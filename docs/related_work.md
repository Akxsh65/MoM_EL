# Related work (T0)

Status: first pass from web search on 2026-11-27. Rows marked **[verify]** come from search snippets or
memory and need a manual read (Scholar / ISMIR archive) before they are cited in the paper.
Overlap: none / partial / full. "Boundaries over time" = evaluates change points / boundaries within a recording.

| # | Citation | Link | Task | Data | Boundaries over time? | Overlap |
|---|---|---|---|---|---|---|
| 1 | Koduri, Gulati, Rao, Serra. Raga recognition based on pitch distribution methods. JNMR 2012 | (base paper; cite from PLAN) | Whole-recording raga recognition (pitch-class distributions, k-NN) | CompMusic Carnatic/Hindustani | No (§6.2 names ragamalika as open) | none (our base) |
| 2 | Gulati, Serrà, Ganguli, Şentürk, Serra. Time-delayed melody surfaces for raga recognition. ISMIR 2016 | https://archives.ismir.net/ismir2016/paper/000030.pdf | Whole-recording recognition with TDMS feature; 87% on 480-rec/40-raga Carnatic set | CMD Carnatic (480 rec, 40 ragas) | No | none (we reuse TDMS as W2 per window) |
| 3 | Chordia & Rae. Real-time raag recognition for interactive music. NIME 2008 **[verify]** | — | Online raga recognition for interaction | Small Hindustani set | Online, but no boundary evaluation **[verify]** | partial |
| 4 | Rāgawise (real-time raga recognition demo) **[verify exact citation]** | — | Real-time raga recognition | — | **[verify]** | partial |
| 5 | DeepSRGM: sequence classification and ranking in Indian classical music with deep learning. arXiv 2402.10168 | https://arxiv.org/pdf/2402.10168 | LSTM raga classification (88.1% CMD Carnatic, 97% 10-raga subset) | CMD | No (clip/recording level) | partial (W3-like model) |
| 6 | Sarala & Murthy. Segmentation of continuous Carnatic concert recordings (applause + raga-based merging). ISMIR 2013 | https://archives.ismir.net/ismir2013/paper/000167.pdf | Item segmentation of concerts; raga classification used to merge segments | Carnatic concerts | Yes, item boundaries (applause-driven), not within-item raga change | partial |
| 7 | Padi & Murthy. Segmentation of continuous audio recordings of Carnatic music concerts into items. Sādhanā 43:154, 2018 | https://link.springer.com/article/10.1007/s12046-018-0922-y | Concert → items | Carnatic concerts | Yes, item boundaries | partial (B2 resembles their target) |
| 8 | Kulkarni. Automatic audio segmentation of Carnatic music. MSc thesis, Erlangen, 2014 | https://www.audiolabs-erlangen.com/content/05_fau/professor/00_mueller/01_group/2014_KulkarniVenkatesh_CarnaticMusic_MasterThesis.pdf | Structural segmentation with tempo/melodic features | Carnatic | Yes, structure, not raga identity | partial |
| 9 | Carnatic raga identification using rigorous time-delay neural network. arXiv 2405.16000 | https://arxiv.org/pdf/2405.16000 | TDNN+LSTM+attention raga ID | Carnatic | No | none |
| 10 | A comparison of audio preprocessing techniques and deep learning algorithms for raga recognition. arXiv 2212.05335 | https://arxiv.org/pdf/2212.05335 | CNN/ANN/LSTM raga ID | Carnatic | No | none |
| 11 | Segmentation of Hindustani alap sections (TISMIR article 64) **[verify citation]** | https://account.transactions.ismir.net/index.php/up-j-tismir/article/view/64/56 | Section boundaries in alap | Hindustani | Yes, section boundaries, not raga | none |
| 12 | Raga verification via longest common segment sets. ISMIR 2015 **[verify authors]** | https://www.ismir2015.uma.es/articles/172_Paper.pdf | Raga verification (phrase matching) | Carnatic/Hindustani | No | none |
| 13 | Koduri, Gulati, Rao. A survey of raaga recognition techniques and improvements to the state of the art. SMC 2011 | (typeset.io copy) | Survey | — | No | none |

## Findings
- No hit performs windowed raga recognition + temporal decoding + boundary evaluation of raga changes. Closest are items 3, 4 (online systems without boundary metrics **[verify]**) and 6, 7 (item segmentation using applause, assuming one raga per item).
- No row is "full overlap". Per T0 backup 1, record differences in decisions.md.
- TODO before paper: read rows 3, 4, 11, 12 fully; extra searches for "raga change point", "melakarta/ragamalika identification" and Hindustani transition work.
