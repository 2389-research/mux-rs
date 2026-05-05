// ABOUTME: FFI-safe mirrors of mux::llm media types for UniFFI bindings.
// ABOUTME: Converts to/from core mux types at the FFI boundary.

use mux::llm::{MediaKind, MediaSource};
use std::path::PathBuf;

/// Kind of media payload — mirror of `mux::llm::MediaKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiMediaKind {
    Image,
    Document,
    Audio,
    Video,
}

/// Source of media bytes — mirror of `mux::llm::MediaSource`.
/// `Path` carries a `String` here because UniFFI does not support `PathBuf`.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiMediaSource {
    Base64 { data: String },
    Url { url: String },
    Path { path: String },
}

/// A media attachment — mirror of `mux::llm::ContentBlock::Media`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMedia {
    pub kind: FfiMediaKind,
    pub source: FfiMediaSource,
    pub mime_type: String,
}

// ---------- From conversions ----------

impl From<FfiMediaKind> for MediaKind {
    fn from(k: FfiMediaKind) -> Self {
        match k {
            FfiMediaKind::Image => MediaKind::Image,
            FfiMediaKind::Document => MediaKind::Document,
            FfiMediaKind::Audio => MediaKind::Audio,
            FfiMediaKind::Video => MediaKind::Video,
        }
    }
}

impl From<MediaKind> for FfiMediaKind {
    fn from(k: MediaKind) -> Self {
        match k {
            MediaKind::Image => FfiMediaKind::Image,
            MediaKind::Document => FfiMediaKind::Document,
            MediaKind::Audio => FfiMediaKind::Audio,
            MediaKind::Video => FfiMediaKind::Video,
        }
    }
}

impl From<FfiMediaSource> for MediaSource {
    fn from(s: FfiMediaSource) -> Self {
        match s {
            FfiMediaSource::Base64 { data } => MediaSource::Base64(data),
            FfiMediaSource::Url { url } => MediaSource::Url(url),
            FfiMediaSource::Path { path } => MediaSource::Path(PathBuf::from(path)),
        }
    }
}

impl From<MediaSource> for FfiMediaSource {
    fn from(s: MediaSource) -> Self {
        match s {
            MediaSource::Base64(data) => FfiMediaSource::Base64 { data },
            MediaSource::Url(url) => FfiMediaSource::Url { url },
            MediaSource::Path(p) => FfiMediaSource::Path {
                path: p.to_string_lossy().into_owned(),
            },
        }
    }
}

impl FfiMedia {
    /// Convert to a `ContentBlock::Media` core type.
    pub fn into_content_block(self) -> mux::llm::ContentBlock {
        mux::llm::ContentBlock::Media {
            kind: self.kind.into(),
            source: self.source.into(),
            mime_type: self.mime_type,
        }
    }

    /// Create an `FfiMedia` from a matching `ContentBlock::Media`. Returns
    /// `None` for any other `ContentBlock` variant.
    pub fn from_content_block(block: &mux::llm::ContentBlock) -> Option<Self> {
        if let mux::llm::ContentBlock::Media {
            kind,
            source,
            mime_type,
        } = block
        {
            Some(FfiMedia {
                kind: (*kind).into(),
                source: source.clone().into(),
                mime_type: mime_type.clone(),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_roundtrip() {
        let all = [
            FfiMediaKind::Image,
            FfiMediaKind::Document,
            FfiMediaKind::Audio,
            FfiMediaKind::Video,
        ];
        for k in all {
            let core: MediaKind = k.into();
            let back: FfiMediaKind = core.into();
            assert_eq!(k, back);
        }
    }

    #[test]
    fn test_source_roundtrip_base64() {
        let ffi = FfiMediaSource::Base64 {
            data: "aGVsbG8=".to_string(),
        };
        let core: MediaSource = ffi.into();
        match &core {
            MediaSource::Base64(s) => assert_eq!(s, "aGVsbG8="),
            _ => panic!(),
        }
        let back: FfiMediaSource = core.into();
        match back {
            FfiMediaSource::Base64 { data } => assert_eq!(data, "aGVsbG8="),
            _ => panic!(),
        }
    }

    #[test]
    fn test_source_roundtrip_path() {
        let ffi = FfiMediaSource::Path {
            path: "/tmp/x.png".to_string(),
        };
        let core: MediaSource = ffi.into();
        match &core {
            MediaSource::Path(p) => assert_eq!(p.to_string_lossy(), "/tmp/x.png"),
            _ => panic!(),
        }
        let back: FfiMediaSource = core.into();
        match back {
            FfiMediaSource::Path { path } => assert_eq!(path, "/tmp/x.png"),
            _ => panic!(),
        }
    }

    #[test]
    fn test_ffi_media_into_content_block() {
        let ffi = FfiMedia {
            kind: FfiMediaKind::Image,
            source: FfiMediaSource::Base64 {
                data: "xxx".to_string(),
            },
            mime_type: "image/png".to_string(),
        };
        let block = ffi.into_content_block();
        match block {
            mux::llm::ContentBlock::Media {
                kind,
                source,
                mime_type,
            } => {
                assert_eq!(kind, MediaKind::Image);
                assert!(matches!(source, MediaSource::Base64(ref s) if s == "xxx"));
                assert_eq!(mime_type, "image/png");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_ffi_media_from_non_media_block_returns_none() {
        let block = mux::llm::ContentBlock::text("hi");
        assert!(FfiMedia::from_content_block(&block).is_none());
    }

    #[test]
    fn test_ffi_media_from_media_block_roundtrip() {
        let original = FfiMedia {
            kind: FfiMediaKind::Audio,
            source: FfiMediaSource::Url {
                url: "https://a.b/x.wav".to_string(),
            },
            mime_type: "audio/wav".to_string(),
        };
        let block = original.clone().into_content_block();
        let back = FfiMedia::from_content_block(&block).unwrap();
        assert_eq!(back.kind, FfiMediaKind::Audio);
        assert_eq!(back.mime_type, "audio/wav");
        match back.source {
            FfiMediaSource::Url { url } => assert_eq!(url, "https://a.b/x.wav"),
            _ => panic!(),
        }
    }
}
