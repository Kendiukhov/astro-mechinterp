# PLOS ONE submission checklist

Submit at **https://journals.plos.org/plosone/s/submit-now** → Editorial Manager.
Article type: **Research Article**.

---

## Publication fee

The PLOS ONE fee is **US $2,477** for a research article, charged at the rate in force on the
*submission* date and payable on acceptance.

You have said the fee is covered, so Publication Fee Assistance is not needed. Two things are
still worth doing in the **Additional Information -> Publication fees** step:

- **Check whether Tubingen has a PLOS institutional agreement.** Many German universities are
  covered by read-and-publish arrangements; if yours is, the fee is settled centrally and you
  pay nothing personally. Selecting the institution in that step is what triggers it.
- Whatever route applies, note that PFA cannot be applied for later: *"Requests made during the
  review process or after acceptance will not be considered."* If there is any chance the
  funding falls through, apply at submission rather than relying on fixing it afterwards.

**Do not mention fees in the cover letter.** PLOS: *"IMPORTANT: Do not include requests to
reduce or waive publication fees in the cover letter."* The supplied cover letter does not.

## Files to upload

| File | Upload as | Notes |
|---|---|---|
| `main.pdf` | Manuscript | Text, tables, figure captions, references. **No figures inside.** |
| `Fig1.tif` … `Fig6.tif` | Figure 1 … Figure 6 | separate items, in `figures/` |
| `S1_File.zip` | Supporting Information | per-experiment results + claim-to-code map |
| `cover-letter.txt` | Cover letter | paste or attach |

LaTeX source (`main.tex`, `refs.bib`, `plos2025.bst`) is **not** uploaded now. PLOS:
*"If your submission is accepted, you will be asked to upload your LaTeX source files."*

---

## Statements to paste into the form

These go in the **submission system only**. PLOS states twice that funding must not appear
anywhere in the manuscript file, and the manuscript has been built accordingly.

**Financial Disclosure** — PLOS's own boilerplate for unfunded work:

> The author(s) received no specific funding for this work.

**Competing Interests** — PLOS's standard no-interest sentence:

> The authors have declared that no competing interests exist.

**Data Availability** — note that a bare GitHub URL does *not* satisfy PLOS, which requires a
repository name plus a persistent identifier. Supporting Information supplies that, because
PLOS deposits SI to figshare at publication and mints the identifier for you:

> All result files underlying every figure, table and numeric claim in this article are
> provided as Supporting Information (S1 File), which includes the recorded JSON output of
> every experiment and a table mapping each numbered claim to the script and key path that
> produced it. The analysis code is openly available at
> https://github.com/Kendiukhov/astro-mechinterp. The underlying survey data are public: the
> PROVABGS value-added catalogue, DESI Early Data Release spectra, DESI Legacy Imaging Surveys
> DR9 imaging and segmentation, and one shard of MultimodalUniverse HSC PDR3; the repository
> documents the expected on-disk layout and how to rebuild the cross-match.

> **Stronger option, a few minutes' work:** archive the repository to Zenodo (GitHub's release
> hook does this) and add the resulting DOI to the statement. Zenodo is on PLOS's recommended
> repository list; GitHub is not.

**Ethics** — none required. No human subjects, animals, cell lines, field sampling or biosafety
implications; answer "no" to the human/animal questions.

---

## Already handled in the manuscript

- Abstract **296 words** (limit 300), one block, no citations
- Numbered Vancouver citations via `plos2025.bst`; **84 references, all resolve**
- Unnumbered headings; section cross-references rewritten as short phrases
- Line numbers on; PLOS title and author block
- Figures **stripped from the PDF**, captions retained in text, "Fig" never "Figure"
- Figures at 300 dpi, RGB, LZW, flattened; 1601–2201 px wide, all inside PLOS's 789–2250 px
  band and under 10 MB (largest is 0.22 MB)
- Figure text raised to 8–9 pt in Arial, meeting PLOS's "Arial, Times or Symbol, 8–12 point"
- AI disclosure in Methods, carrying all three elements PLOS requires: the tool named, what it
  affected, and how its outputs were validated
- No funding wording anywhere in the manuscript file

## Worth deciding before you submit

**Does this "dispute published work"?** If PLOS classes it so, policy is to invite a signed
review from the disputed authors — the AION-1 team — and you must accept that condition before
the manuscript proceeds. The cover letter pre-empts it by stating plainly that the paper
contradicts no claim AION-1's authors make and instead measures a behaviour they did not
characterise. That is accurate, and it is the honest framing.

**Astronomy is thin at PLOS ONE.** The subject area exists and is occupied, but at roughly one
paper a year (36 astronomy papers in a corpus of ~344,000), and there is no astronomy section
editor — the nearest are "Physics and astronomy — Multidisciplinary" and "Artificial
intelligence, machine learning and data science". The ML side of the paper sits in a body of
thousands. Expect the handling editor to come from the ML side.
