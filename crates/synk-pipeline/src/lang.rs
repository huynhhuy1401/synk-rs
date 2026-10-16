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
            .with_next("Use an ISO 639-1 code such as \"vi\" or \"en\"")
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
    whatlang::detect(text)
        .filter(|info| info.confidence() >= 0.5)
        .map(|info| info.lang())
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
