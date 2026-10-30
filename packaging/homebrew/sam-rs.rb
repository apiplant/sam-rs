# Generated from packaging/homebrew/sam-rs.rb in apiplant/sam-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/sam-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class SamRs < Formula
  desc "Rust (candle) inference for Segment Anything models (SAM, SAM 2, SAM 3)"
  homepage "https://github.com/apiplant/sam-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # No bottles: the release archives *are* the binaries, so the formula only
  # unpacks what the tagged workflow already built for each platform.
  on_macos do
    on_arm do
      url "https://github.com/apiplant/sam-rs/releases/download/v@VERSION@/sam-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/sam-rs/releases/download/v@VERSION@/sam-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/sam-rs/releases/download/v@VERSION@/sam-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  conflicts_with "sam-rs-cuda", because: "both install the same binaries"

  def install
    bin.install "sam-rs"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/sam-rs --version")
  end
end
