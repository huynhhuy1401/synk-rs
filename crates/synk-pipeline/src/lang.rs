//! Languages: parsing the ones a user configures, and detecting the
//! language of a chunk or query.

use synk_core::{ErrorCode, Result, SynkError};
use whatlang::Lang;

/// ISO 639-1 codes for whatlang's languages, so users can write `vi`.
const ISO_639_1: &[(&str, Lang)] = &[
    ("af", Lang::Afr),
    ("ar", Lang::Ara),
    ("az", Lang::Aze),
    ("be", Lang::Bel),
    ("bg", Lang::Bul),
    ("bn", Lang::Ben),
    ("ca", Lang::Cat),
    ("cs", Lang::Ces),
    ("da", Lang::Dan),
    ("de", Lang::Deu),
    ("el", Lang::Ell),
    ("en", Lang::Eng),
    ("eo", Lang::Epo),
    ("es", Lang::Spa),
    ("et", Lang::Est),
    ("fa", Lang::Pes),
    ("fi", Lang::Fin),
    ("fr", Lang::Fra),
    ("gu", Lang::Guj),
    ("he", Lang::Heb),
    ("hi", Lang::Hin),
    ("hr", Lang::Hrv),
    ("hu", Lang::Hun),
    ("hy", Lang::Hye),
    ("id", Lang::Ind),
    ("it", Lang::Ita),
    ("ja", Lang::Jpn),
    ("jv", Lang::Jav),
    ("ka", Lang::Kat),
    ("km", Lang::Khm),
    ("kn", Lang::Kan),
    ("ko", Lang::Kor),
    ("la", Lang::Lat),
    ("lt", Lang::Lit),
    ("lv", Lang::Lav),
    ("mk", Lang::Mkd),
    ("ml", Lang::Mal),
    ("mr", Lang::Mar),
    ("my", Lang::Mya),
    ("ne", Lang::Nep),
    ("nl", Lang::Nld),
    ("nb", Lang::Nob),
    ("pa", Lang::Pan),
    ("pl", Lang::Pol),
    ("pt", Lang::Por),
    ("ro", Lang::Ron),
    ("ru", Lang::Rus),
    ("si", Lang::Sin),
    ("sk", Lang::Slk),
    ("sl", Lang::Slv),
    ("sn", Lang::Sna),
    ("sr", Lang::Srp),
    ("sv", Lang::Swe),
    ("ta", Lang::Tam),
    ("te", Lang::Tel),
    ("th", Lang::Tha),
    ("tk", Lang::Tuk),
    ("tl", Lang::Tgl),
    ("tr", Lang::Tur),
    ("uk", Lang::Ukr),
    ("ur", Lang::Urd),
    ("uz", Lang::Uzb),
    ("vi", Lang::Vie),
    ("yi", Lang::Yid),
    ("zh", Lang::Cmn),
    ("zu", Lang::Zul),
];

/// Parses an ISO 639-1 code (`vi`), an ISO 639-3 code (`vie`) or an English
/// name (`Vietnamese`).
pub fn parse(s: &str) -> Result<Lang> {
    let lower = s.trim().to_lowercase();
    ISO_639_1
        .iter()
        .find(|(code, _)| *code == lower)
        .map(|(_, lang)| *lang)
        .or_else(|| Lang::from_code(lower.as_str()))
        .or_else(|| {
            Lang::all()
                .iter()
                .copied()
                .find(|l| l.eng_name().eq_ignore_ascii_case(&lower))
        })
        .ok_or_else(|| {
            SynkError::new(
                ErrorCode::ConfigInvalid,
                format!("unknown language `{s}` in [generator] languages"),
            )
            .with_hint("use an ISO 639-1 code such as \"vi\" or \"en\"")
        })
}

/// Short code for reports: ISO 639-1 when there is one, else ISO 639-3.
pub fn code(lang: Lang) -> &'static str {
    ISO_639_1
        .iter()
        .find(|(_, l)| *l == lang)
        .map_or(lang.code(), |(code, _)| code)
}

/// The language of a text, if it can be told with reasonable confidence.
pub fn detect(text: &str) -> Option<Lang> {
    if vietnamese_letters(text) {
        return Some(Lang::Vie);
    }
    detect_general(text)
}

/// whatlang alone: what `detect` returned before the Vietnamese rule.
pub fn detect_general(text: &str) -> Option<Lang> {
    whatlang::detect(text)
        .filter(|info| info.confidence() >= 0.5)
        .map(|info| info.lang())
}

/// Share of letters that only Vietnamese uses, above which a text is
/// Vietnamese. Notes that mix Vietnamese with English terms
/// ("deploy lên server bị lỗi timeout") fool general detectors, which
/// called such lines French or Latvian; Vietnamese's own letters don't.
const VIETNAMESE_LETTER_SHARE: f64 = 0.02;

/// Whether at least [`VIETNAMESE_LETTER_SHARE`] of the letters are ones
/// only Vietnamese uses: ă, đ, ơ, ư, ĩ, ũ, and the stacked tone-mark
/// vowels (U+1EA0..U+1EF9), counted after NFC so decomposed text counts
/// too. â, ê, ô and plain accents are shared with French and others.
fn vietnamese_letters(text: &str) -> bool {
    use unicode_normalization::UnicodeNormalization;
    let (mut letters, mut vietnamese) = (0u32, 0u32);
    for c in text.nfc() {
        if !c.is_alphabetic() {
            continue;
        }
        letters += 1;
        let lower = c.to_lowercase().next().unwrap_or(c);
        if matches!(lower, 'ă' | 'đ' | 'ơ' | 'ư' | 'ĩ' | 'ũ')
            || ('\u{1EA0}'..='\u{1EF9}').contains(&c)
        {
            vietnamese += 1;
        }
    }
    letters > 0 && f64::from(vietnamese) / f64::from(letters) >= VIETNAMESE_LETTER_SHARE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_codes_and_names() {
        assert_eq!(parse("vi").unwrap(), Lang::Vie);
        assert_eq!(parse("VIE").unwrap(), Lang::Vie);
        assert_eq!(parse("Vietnamese").unwrap(), Lang::Vie);
        assert_eq!(parse("klingon").unwrap_err().code, ErrorCode::ConfigInvalid);
        assert_eq!(code(Lang::Vie), "vi");
    }

    #[test]
    fn mixed_and_decomposed_vietnamese_is_vietnamese() {
        use unicode_normalization::UnicodeNormalization;
        for vi in [
            "Deploy lên server bị lỗi timeout, phải tăng keepalive lên 75s trong nginx.",
            "Tăng max_connections của PostgreSQL lên 500 khi tải cao",
            "Sao lưu database chạy lúc 2 giờ sáng",
        ] {
            assert_eq!(detect(vi), Some(Lang::Vie), "{vi}");
        }
        // Typed or exported as base letters plus combining marks.
        let decomposed: String = "Tiếng Việt có dấu".nfd().collect();
        assert_eq!(detect(&decomposed), Some(Lang::Vie));
        // French and English stay what they are.
        assert_eq!(
            detect("Le serveur était très lent après la mise à jour de la base de données."),
            Some(Lang::Fra)
        );
        // Shared accents (é, â, ê, ô) don't make a text Vietnamese.
        assert_ne!(
            detect("The café's menu changed after the update to the ordering system."),
            Some(Lang::Vie)
        );
    }

    #[test]
    fn detects_vietnamese_and_english() {
        assert_eq!(
            detect("Nhân viên được nghỉ phép 12 ngày mỗi năm, phải báo trước hai tuần."),
            Some(Lang::Vie)
        );
        assert_eq!(
            detect("Backups run nightly at two in the morning and are kept for thirty days."),
            Some(Lang::Eng)
        );
    }
}
