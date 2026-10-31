//! Text-prompted detect-and-track for SAM 3 video.
//!
//! The detector finds every instance of a text concept; each new instance starts a tracker [`Session`] seeded
//! with its box, and the tracker follows it through the following frames. The detector re-runs every
//! `detect_every` frames so objects entering later are picked up; a detection that overlaps a live track is
//! treated as that track, and a track that stays absent for `max_missing` frames is dropped.
//!
//! This is the same detector + tracker pairing the model is built around, but with a simple greedy association
//! rule rather than the reference's full matching logic, so identities can differ from the reference in crowded
//! or heavily occluded scenes.

use anyhow::Result;
use image::RgbImage;

use super::video::{FramePrompt, Session};
use super::{Instance, Sam3};
use crate::Mask;

/// A tracked object on one frame.
pub struct TrackedObject {
    /// Stable id, assigned in order of first detection.
    pub id: usize,
    /// Tracker presence logit (positive means the object is visible).
    pub object_score: f32,
    /// Detector score when the object was detected on this frame.
    pub detection_score: Option<f32>,
    /// The concept ("wheel") this object was found for.
    pub label: String,
    pub mask: Mask,
}

struct Track {
    id: usize,
    label: String,
    session: Session,
    missing: usize,
}

/// The tracker's state without a borrow of the model, so it can live next to the model in a server.
/// [`TextTracker`] is the convenient wrapper.
pub struct TextState {
    /// the concepts to find, e.g. `["wheel", "person in red"]`
    concepts: Vec<String>,
    threshold: f32,
    detect_every: usize,
    max_missing: usize,
    /// Minimum mask IoU for a detection to count as an existing track.
    match_iou: f32,
    tracks: Vec<Track>,
    next_id: usize,
    frame: usize,
}

/// Splits a prompt into concepts at full stops, commas, semicolons and new lines.
pub fn concepts(prompt: &str) -> Vec<String> {
    prompt.split(['.', ',', ';', '\n']).map(str::trim).filter(|c| !c.is_empty()).map(str::to_string).collect()
}

fn mask_iou(a: &Mask, b: &Mask) -> f32 {
    if a.data.len() != b.data.len() {
        return 0.0;
    }
    let (mut inter, mut union) = (0usize, 0usize);
    for (&x, &y) in a.data.iter().zip(&b.data) {
        inter += (x > 0 && y > 0) as usize;
        union += (x > 0 || y > 0) as usize;
    }
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

impl TextState {
    /// `prompt`: one concept, or several separated by full stops, commas, semicolons or new lines ("wheel. person in red").
    pub fn new(prompt: &str, threshold: f32) -> Self {
        Self { concepts: concepts(prompt), threshold, detect_every: 5, max_missing: 8, match_iou: 0.5, tracks: vec![], next_id: 0, frame: 0 }
    }

    /// Re-run the detector every `n` frames (1 = every frame; the first frame is always detected).
    pub fn detect_every(mut self, n: usize) -> Self {
        self.detect_every = n.max(1);
        self
    }

    /// Drop a track after it has been absent for this many consecutive frames.
    pub fn max_missing(mut self, n: usize) -> Self {
        self.max_missing = n;
        self
    }

    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Processes the next frame and returns every live object on it (absent objects are omitted).
    pub fn step(&mut self, sam: &Sam3, img: &RgbImage) -> Result<Vec<TrackedObject>> {
        let detect = self.frame % self.detect_every == 0;
        let orig = (img.height() as usize, img.width() as usize);
        let (tf, detections) = if detect {
            let f = sam.encode_image(img)?;
            // every concept is detected on its own; the strongest detections claim an object first
            let mut det: Vec<(String, Instance)> = vec![];
            for c in &self.concepts {
                det.extend(sam.segment(&f, c, self.threshold)?.into_iter().map(|i| (c.clone(), i)));
            }
            det.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
            (sam.tracker_features(&f)?, det)
        } else {
            (sam.encode_for_tracking(img)?, vec![])
        };

        // advance the existing tracks
        let mut out: Vec<TrackedObject> = vec![];
        for t in &mut self.tracks {
            let r = sam.track_frame(&mut t.session, &tf)?;
            if r.object_score > 0.0 {
                let mask = sam.frame_mask(&r, orig)?;
                t.missing = 0;
                out.push(TrackedObject { id: t.id, object_score: r.object_score, detection_score: None, label: t.label.clone(), mask });
            } else {
                t.missing += 1;
            }
        }

        // new detections that no live track explains start new tracks
        let (sx, sy) = (super::SIZE as f32 / orig.1 as f32, super::SIZE as f32 / orig.0 as f32);
        for (label, Instance { score, bbox, mask }) in detections {
            if let Some(o) = out.iter_mut().find(|o| mask_iou(&o.mask, &mask) >= self.match_iou) {
                o.detection_score = Some(score);
                continue;
            }
            let mut session = Session::new(None);
            session.add_prompt(0, FramePrompt { points: vec![(bbox[0] * sx, bbox[1] * sy, 2), (bbox[2] * sx, bbox[3] * sy, 3)] });
            let r = sam.track_frame(&mut session, &tf)?;
            let tracked = sam.frame_mask(&r, orig)?;
            // the box-seeded mask is used unless the tracker rejects the box; the detector's own mask is the fallback
            let (m, os) = if r.object_score > 0.0 { (tracked, r.object_score) } else { (mask, r.object_score) };
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.push(Track { id, label: label.clone(), session, missing: 0 });
            out.push(TrackedObject { id, object_score: os, detection_score: Some(score), label, mask: m });
        }

        let limit = self.max_missing;
        self.tracks.retain(|t| t.missing <= limit);
        self.frame += 1;
        out.sort_by_key(|o| o.id);
        Ok(out)
    }
}

/// A [`TextState`] bound to a model.
pub struct TextTracker<'a> {
    sam: &'a Sam3,
    state: TextState,
}

impl<'a> TextTracker<'a> {
    pub fn new(sam: &'a Sam3, prompt: &str, threshold: f32) -> Self {
        Self { sam, state: TextState::new(prompt, threshold) }
    }

    pub fn detect_every(mut self, n: usize) -> Self {
        self.state = self.state.detect_every(n);
        self
    }

    pub fn max_missing(mut self, n: usize) -> Self {
        self.state = self.state.max_missing(n);
        self
    }

    pub fn num_tracks(&self) -> usize {
        self.state.num_tracks()
    }

    /// Processes the next frame and returns every live object on it.
    pub fn step(&mut self, img: &RgbImage) -> Result<Vec<TrackedObject>> {
        self.state.step(self.sam, img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(data: Vec<u8>) -> Mask {
        Mask { width: data.len(), height: 1, data, iou: 1.0 }
    }

    #[test]
    fn splits_concepts() {
        assert_eq!(concepts("wheel. person in red, dog;\ncat"), vec!["wheel", "person in red", "dog", "cat"]);
        assert_eq!(concepts("  truck  "), vec!["truck"]);
        assert!(concepts(" . , ").is_empty());
    }

    #[test]
    fn iou_of_binary_masks() {
        assert_eq!(mask_iou(&m(vec![1, 1, 0, 0]), &m(vec![1, 1, 0, 0])), 1.0);
        assert_eq!(mask_iou(&m(vec![1, 1, 0, 0]), &m(vec![0, 0, 1, 1])), 0.0);
        assert!((mask_iou(&m(vec![1, 1, 1, 0]), &m(vec![0, 1, 1, 1])) - 0.5).abs() < 1e-6);
        assert_eq!(mask_iou(&m(vec![0, 0]), &m(vec![0, 0])), 0.0);
        assert_eq!(mask_iou(&m(vec![1]), &m(vec![1, 1])), 0.0);
    }
}
