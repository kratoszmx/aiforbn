# Three source-to-record cards

1. **Kim et al., 2022, raw BNNT PP at 0.3 mg/cm².**
   [Original article](https://doi.org/10.3390/nano12010011), Methods §2.3 and
   Results §3. Recipe paragraph specifies BNNT 40 mg + PVDF 10 mg in 2 mL NMP,
   one-hour sonication, overnight stirring, 50°C vacuum drying for 24 h. The results
   text reports 0.71 mS/cm versus 0.43 for bare PP. Record:
   `kim_2022:BNNT-PP-0.3`; evidence `kim_2022:recipe`, `kim_2022:conductivity`.
   Recalculation: `(0.71/0.43 − 1) × 100 = 65.1163%`. This is a reconstruction
   of published material performance, not an improvement caused by our model.

2. **Tian et al., 2024, CA@BN-100 and a failed binder ratio.**
   [Original article](https://doi.org/10.3390/molecules29225311), §§2.1, 2.2, 4.2.2.
   A 100 µm applicator produces a reported 110 µm final separator; these fields are
   separated. At 65°C, CA@BN-100 gives 2.80 mS/cm versus 1.28 for the paper's PP
   control. The preparation paragraph reports pore clogging/failure at BN:PVDF 3:1,
   then uses 4:1. The failed formulation remains a separate partial record with
   unknown applicator geometry. Coating sides and drying duration stay missing.
   CA values are excluded from the PP prediction cohort.

3. **Hong et al., 2026, CNF/BNNT loading and water contact angle.**
   [Original article](https://doi.org/10.3390/en19071600), §2.2 and Table 2.
   HCNF, HBNT-05 and HBNT-10 correspond to 0, 0.5 and 1.0 reported wt% BNNT;
   the denominator is not asserted. Water contact angles are 28.45 ± 3.34,
   50.23 ± 3.14 and 68.54 ± 1.55 degrees. This trend is not described as
   improved electrolyte wettability. These are three formulations with reported
   error terms, not nine samples or ionic-conductivity labels.

Each card can be inspected in the website's recipe list by opening the original
evidence paragraph. The dataset loader verifies those paragraphs against the
retained licensed source XML.
