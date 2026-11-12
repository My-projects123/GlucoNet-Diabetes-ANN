"""Build the final project report (DOCX + PDF via Pages).

Two passes: the first pass exports a PDF to learn on which page every heading,
figure, and table lands; the second pass writes those page numbers into the
table of contents and the lists of figures and tables.

Usage (needs Pages for the PDF step):
    cd diabetes_ann && .venv/bin/python final_report/build_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import fitz
import numpy as np
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from diabetes_prediction.config import FEATURE_COLUMNS, ZERO_AS_MISSING_COLUMNS  # noqa: E402
from diabetes_prediction.data import load_dataset  # noqa: E402
from references import CATALOGUE  # noqa: E402
from report_lib import CENTER, LEFT, TITLE_PT, References, Report, add_runs  # noqa: E402

FIG = ROOT / "figures"
RESULTS = ROOT / "results"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
DOCX_OUT = ROOT / "Diabetes_Prediction_ANN_Project_Report.docx"
PDF_OUT = ROOT / "Diabetes_Prediction_ANN_Project_Report.pdf"
LOGO = ROOT / "assets" / "srm_logo.jpeg"

RES = json.loads((RESULTS / "final_results.json").read_text(encoding="utf-8"))
EDA = json.loads((RESULTS / "eda_summary.json").read_text(encoding="utf-8"))
DF = load_dataset()

PROPOSED = "Proposed MLP ANN (+ROS)"
METRIC_ORDER = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
METRIC_NAME = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
SHORT = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "BloodPressure",
    "SkinThickness": "SkinThickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Pedigree (DPF)",
    "Age": "Age",
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def ms(summary: dict, metric: str) -> str:
    return f"{summary[metric]['mean']:.3f} ± {summary[metric]['std']:.3f}"


def rep(name: str, metric: str) -> float:
    return RES["repeated_cv"][name]["summary"][metric]["mean"]


def rep_sd(name: str, metric: str) -> float:
    return RES["repeated_cv"][name]["summary"][metric]["std"]


def abl(group: str, label: str, metric: str) -> float:
    return RES["ablation"][f"{group}::{label}"]["summary"][metric]["mean"]


def leak(key: str, metric: str) -> float:
    return RES["leakage"][key][metric]["mean"]


def ttest(other: str, metric: str) -> dict:
    return RES["corrected_t_tests"][other][metric]


def pval(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def tcell(t: dict) -> str:
    p = "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
    return f"{t['mean_diff']:+.3f} ({p})"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-–'%±/]*", re.sub(r"\*", "", text)))


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def build(page_map: dict[str, int]) -> Report:
    refs = References(CATALOGUE)
    c = refs.cite
    rpt = Report(BUILD, page_map, refs)
    ho = RES["holdout"]["models"]
    ho_p = ho[PROPOSED]
    cm_p = ho_p["confusion_matrix"]
    tn, fp, fn, tp = cm_p[0][0], cm_p[0][1], cm_p[1][0], cm_p[1][1]
    nested = RES["nested"]
    seed = RES["seed_stability"]["summary"]
    zeros = (DF[FEATURE_COLUMNS] == 0).sum()

    # ------------------------------------------------------------ title page
    if LOGO.exists():
        par = rpt.doc.add_paragraph()
        par.alignment = CENTER
        par.add_run().add_picture(str(LOGO), width=Inches(2.6))
    rpt.para("", space_after=18)
    rpt.para("DIABETES PREDICTION SYSTEM USING A LEAKAGE-SAFE ARTIFICIAL NEURAL NETWORK",
             align=CENTER, size=TITLE_PT, bold=True, space_after=18)
    rpt.para("A PROJECT REPORT", align=CENTER, bold=True, space_after=6)
    rpt.para("submitted for the course", align=CENTER, italic=True, space_after=2)
    rpt.para("21CSE326T – ARTIFICIAL NEURAL NETWORKS", align=CENTER, bold=True, space_after=24)
    rpt.para("Submitted by", align=CENTER, italic=True, space_after=6)
    rpt.para("SHANMUGAPRIYAN J (RA2411026050011)", align=CENTER, bold=True, space_after=2)
    rpt.para("DEEPAK V F (RA2411026050015)", align=CENTER, bold=True, space_after=24)
    rpt.para("III Year / V Semester, Section A", align=CENTER, space_after=2)
    rpt.para("B.Tech. Computer Science and Engineering with Specialization in Artificial Intelligence and "
             "Machine Learning", align=CENTER, space_after=36)
    for line in ["SCHOOL OF COMPUTING", "FACULTY OF ENGINEERING AND TECHNOLOGY",
                 "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", "TIRUCHIRAPPALLI – 621105"]:
        rpt.para(line, align=CENTER, bold=True, space_after=2)
    rpt.para("OCTOBER 2026", align=CENTER, bold=True, space_before=18)

    # -------------------------------------------------------------- abstract
    rpt.unnumbered_heading("ABSTRACT", toc=False, center=True, new_page=True).paragraph_format.space_after = Pt(4)
    abstract = [
        "According to the World Health Organization, at present, about 830 million people in the world are "
        "living with diabetes mellitus. The scope of this course is narrower: can a handful of easily "
        "affordable clinical measures point to which patients are most likely to test positive? The Pima "
        "Indians diabetes dataset is the conventional working standard for that task. It was compiled by the "
        "National Institute of Diabetes and Digestive and Kidney Diseases from adult Pima women near Phoenix, "
        "Arizona. In 1988, Smith et al. had already trained an initial neural model on the collection. UCI "
        "still stores it with a total of 768 records. The variables included in the record indicate the number "
        "of pregnancies, 2-hour plasma glucose, diastolic blood pressure, triceps skinfold thickness, 2-hour "
        "serum insulin, BMI, pedigree score, and age.",
        "The published accuracies in the present document fluctuate considerably. There has been information "
        "from studies whose recorded accuracy is approximately 76% and others that exceeded 90%. We found "
        "that most of those papers were based on the same protocol. The preprocessing was applied to all of "
        "the 768 entries before any train–test separation. Zeros in glucose or insulin were counted as real "
        "results, even though these values would be impossible, given the physiology of an organism. Although "
        "accuracy is often treated as the leading criteria, a sizeable portion (65.1%) of the entries contains "
        "negative cases.",
        "In this work, a small feed-forward artificial neural network has been trained while avoiding the use "
        "of shortcuts. Impossible zeros for glucose, blood pressure, thickness of skin, insulin levels and BMI "
        "have been changed into missing values; a tally of zero pregnancies remains unchanged. The median "
        "imputation and standardisation have been estimated only from the training fold. Oversampling of the "
        "minority class has been carried out in the training fold, and then an 8–16–8–1 neural network with "
        "289 weights has been trained using Adam, small L2 penalty and early stopping.",
        "In the same folds, logistic regression, random forest and dummy classifier were evaluated. The values "
        "of accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC and confusion matrices are "
        "reported. In 10 x 5 repeated cross-validation, the network achieved sensitivity of 0.768, specificity "
        "of 0.734, F1 of 0.679, and ROC-AUC of 0.834. In the holdout test comprising 154 records 43 out of 54 "
        "diabetes patients were identified correctly (ROC-AUC of 0.821). Logistic regression (ROC-AUC of 0.836) "
        "and random forest (ROC-AUC of 0.838) scored marginally higher than the developed network. "
        "Oversampling in the training fold increased sensitivity by 0.208 (p < 0.001). The elimination of "
        "pipeline elements resulted in a fall of ROC-AUC to 0.721 owing to the absence of standardisation. In "
        "the same manner, the elimination of early stopping led to a reduction of the ROC-AUC to 0.784, and "
        "the removal of medical zeros ruled ROC-AUC dropped the value to 0.820. Achieveing deeper models "
        "turned out to have a minor influence, as well as activation choice or L2 regularisation applied, as "
        "well as univariate feature selection.",
        "The oversampling performed before the cross-validation raised the value of PR-AUC from 0.706 to "
        "0.825. In contrast to that finding, the nested cross-validation showed drop of the best tuned "
        "ROC-AUC from 0.835 to 0.819.",
        "Ten different random seeds resulted in a ROC-AUC fluctuation value around 0.04, which is in line "
        "with most architectural changes.",
        "The foremost feature that appears to have a significant impact on accuracy is glucose, being "
        "followed by age, BMI and number of pregnancies.",
    ]
    for p in abstract:
        rpt.para(p, space_after=2)
    rpt.para("**Keywords:** artificial neural network, multilayer perceptron, diabetes prediction, Pima Indians "
             "Diabetes dataset, data leakage, class imbalance, ablation study.", space_before=2, space_after=0)
    abstract_words = sum(word_count(p) for p in abstract)

    # ------------------------------------------------------- listings anchors
    rpt.unnumbered_heading("TABLE OF CONTENTS", toc=False, center=True, new_page=True)
    toc_anchor = rpt.anchor()
    rpt.unnumbered_heading("LIST OF FIGURES", toc=False, center=True, new_page=False)
    lof_anchor = rpt.anchor()
    rpt.unnumbered_heading("LIST OF TABLES", toc=False, center=True, new_page=False)
    lot_anchor = rpt.anchor()

    rpt.new_numbered_section()

    # ============================================================ CHAPTER 1
    rpt.chapter_heading("Introduction", new_page=False)
    rpt.section("1.1 Background")
    rpt.para(
        "Diabetes mellitus is not one disease. The body either makes too little insulin or cannot use the "
        "insulin it has. Glucose stays high. Over years that eats into vessels, kidneys, eyes and nerves. WHO "
        f"lists it among the main drivers of heart attack, stroke, kidney failure, blindness and amputation "
        f"{c('who_fact')}. Adult cases went from about 200 million in 1990 to about 830 million in 2022. "
        f"Prevalence roughly doubled, 7% to 14% {c('who_fact', 'who_2024')}. IDF thinks the curve keeps climbing "
        f"through 2045 {c('idf2021')}. Plenty of type 2 cases are still missed. A cheap flag from routine "
        "numbers will not diagnose anyone. It can still decide who gets tested first."
    )
    rpt.para(
        "Clinic diagnosis still uses lab cut-offs. Fasting plasma glucose ≥ 126 mg/dL. Two-hour OGTT plasma "
        f"glucose ≥ 200 mg/dL. Or HbA1c ≥ 6.5% {c('ada2024')}. We are not replacing those tests. The job here "
        "is risk ranking: take a handful of measurements and guess how likely a person is to come back "
        "positive, so the scarce tests go where they might matter. Machine learning already sits in a lot of "
        f"diabetes papers {c('kavakiotis2017')}. ANNs show up in a big share of them."
    )
    rpt.para(
        "A feed-forward ANN (multilayer perceptron) pushes an input through weighted sums and nonlinear "
        "activations and ends with a probability. Give it enough hidden units and it can approximate any "
        f"continuous function on a bounded set {c('hornik1989')}. Weights come from gradient descent plus "
        f"back-propagation {c('rumelhart1986')}. That is handy when glucose, BMI and age may interact. It is "
        "also how you over-fit a 768-row table if you are not careful. That tension is the actual project."
    )
    rpt.section("1.2 The Pima Indians Diabetes Benchmark")
    rpt.para(
        "Gila River, near Phoenix, has one of the highest recorded type 2 rates anywhere "
        f"{c('bennett1971')}. Knowler et al. put the incidence at about 19 times Rochester, Minnesota "
        f"{c('knowler1978')}. NIDDK kept examining the community from 1965. In 1988 Smith et al. pulled 768 "
        "exams of women aged 21 or older and tried to forecast diabetes inside five years with ADAP, an early "
        f"neural routine {c('smith1988')}. UCI got the extract in 1990 {c('uci_pima')}. Coursework has used "
        "it ever since."
    )
    rpt.para(
        "The file is small, all numbers, binary label. Easy for a laptop project. It also has traps. Classes "
        "are off: 500 negatives (65.1%) and 268 positives (34.9%). Always say \"no diabetes\" and you already "
        "hit 65.1% accuracy. Older notes claim no missing values. Then you look at glucose, blood pressure, "
        "skinfold, insulin and BMI. Zeros. A living adult cannot have those. They are missing-value codes "
        f"{c('mlbench', 'ripley1996', 'wahba1995')}. With only 768 rows, a little leak from the test side into "
        "training is enough to puff the scores up."
    )
    rpt.section("1.3 Problem Statement")
    rpt.para(
        "Given eight numbers for an adult Pima woman (pregnancies, 2-hour OGTT glucose, diastolic blood "
        "pressure, triceps skinfold, 2-hour serum insulin, BMI, pedigree, age), estimate P(*y* = 1) under the "
        "study rule and then cut that probability into a class. The model has to be an ANN. Nothing computed "
        "from an evaluation row can leak into fitting. We compare it with simpler baselines on the same "
        "splits. Metrics have to still mean something when 65% of rows are negative."
    )
    rpt.section("1.4 Motivation")
    rpt.para(
        "Leakage is the boring reason a lot of applied papers look stronger than they are: evaluation data "
        f"sneaks into model building {c('kaufman2012', 'kapoor2023')}. On Pima it is usually the imputer, "
        "scaler or selector fitted on all 768 rows before CV. Or oversampling before the split, so copies of "
        "the same woman sit in train and test. Or tuning on the same folds you later quote. Reporting is the "
        "other mess. Bozkurt et al. already said many older Pima ANN papers quoted accuracy only "
        f"{c('bozkurt2014')}, which is a weak summary when classes are skewed {c('saito2015')}. Prediction-model "
        f"checklists also want a clear missing-data story and a real validation {c('collins2015')}. We wanted "
        "the lab task done tightly enough that the numbers hold, and we wanted a number on how much the usual "
        "shortcuts actually move them."
    )
    rpt.section("1.5 Objectives")
    rpt.numbered([
        "Load the dataset from a local file only, profile class balance, duplicates, missing values, and zero "
        "codes, and write down every data decision.",
        "Recode medically impossible zeros as missing in five measurements, while keeping Pregnancies = 0 as a "
        "valid value.",
        "Build a pipeline in which imputation, optional feature selection, scaling, and oversampling are fitted "
        "on the training fold only.",
        "Design and train a feed-forward ANN (8–16–8–1) with ReLU hidden units, a sigmoid output, Adam, L2 "
        "regularisation, and early stopping.",
        "Compare the ANN with a dummy classifier, logistic regression, and a random forest on the same "
        "stratified folds, and test differences with a corrected resampled t-test.",
        "Report accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC, and confusion matrices, "
        "and score the model once on an untouched hold-out set.",
        "Run an ablation of pipeline pieces and ANN hyperparameters, and measure the effect of leakage and of "
        "non-nested hyperparameter selection.",
        "Compare the results with published Pima studies and explain where the protocols differ.",
    ])
    rpt.section("1.6 Scope and Limitations")
    rpt.para(
        "Binary classification on 768 Pima rows. Small MLP plus classical baselines. Laptop CPU. Not a "
        "clinical study. The population is narrow: adult women from one community, 1980s exams, a historical "
        "WHO-style glucose rule, nearly half the insulin values missing, no outside cohort. Do not use these "
        "numbers for diagnosis. Do not push them onto men, children or other groups."
    )
    rpt.section("1.7 Contributions")
    rpt.bullets([
        "A reproducible, leakage-safe ANN pipeline for the Pima benchmark, with a medically justified zero "
        "policy, implemented as a tested Python package (20 unit tests).",
        "A fair comparison of the ANN with three baselines on the same folds, using repeated cross-validation, "
        "corrected significance tests, bootstrap intervals, and a single hold-out evaluation.",
        "An ablation of more than thirty configurations that shows which design choices actually matter for "
        "the ANN.",
        "A controlled leakage experiment that measures how much common protocol shortcuts inflate F1, ROC-AUC, "
        "and PR-AUC, which helps explain why published numbers sit so far apart.",
    ])
    rpt.section("1.8 Organisation of the Report")
    rpt.para(
        "Chapter 2 is the literature: Pima ANN papers, classical models, and the methods work on leakage and "
        "imbalance. It ends with a comparison table and the gap we actually chased. Chapter 3 is the method "
        "(block diagram, equations, algorithms). Chapter 4 is the data and the numbers, with console "
        "screenshots. Chapter 5 is ablation plus published scores. Chapter 6 is the wrap-up and what we would "
        "do next."
    )

    # ============================================================ CHAPTER 2
    rpt.chapter_heading("Literature Survey")
    rpt.para(
        "Four piles of papers. The original Pima modelling study. Later neural nets on the same table. "
        "Classical and ensemble methods. Then the methods papers on evaluation. We quote the numbers as the "
        "authors wrote them. Protocols are not the same from paper to paper. Section 2.6 and Chapter 5 pick "
        "that apart."
    )
    rpt.section("2.1 Foundational Study")
    rpt.para(
        f"Smith et al. {c('smith1988')} are the reason this file exists as a modelling task. They used ADAP, "
        "an adaptive learning routine built from perceptron-like devices, to guess who would develop "
        "non-insulin-dependent diabetes inside five years. The label was not a clinic diagnosis in the modern "
        "sense. It was a 2-hour post-load plasma glucose of at least 200 mg/dL, either at a survey exam or in "
        "routine care. They trained on 576 of the 768 eligible exams and held out 192. Cut the real-valued "
        "output at 0.448 and both sensitivity and specificity landed at 76%. They also drew a full ROC curve. "
        "That last bit still looks unusually honest. A lot of later papers just quoted accuracy and stopped."
    )
    rpt.section("2.2 Neural-Network Approaches on the Pima Dataset")
    rpt.para(
        f"Kayaer and Yıldırım {c('kayaer2003')} tried multilayer perceptrons, radial basis function networks, "
        "and a general regression neural network (GRNN) on the UCI extract. Ordinary train/test split. The "
        "GRNN came out on top at 80.21% test accuracy. An MLP trained with Levenberg–Marquardt sat at 77.08%. "
        "Not a huge gap. Polat and Güneş {0} squeezed the eight features down to four principal components, "
        "classified them with ANFIS, and quoted 89.47%. Kahramanli and Allahverdi {1} mixed a crisp ANN with "
        "a fuzzy net trained by back-propagation. Their k-fold figure was 84.24%. Different recipes, different "
        "headlines. Lining them up is already a bit of a stretch.".format(c("polat2007"), c("kahramanli2008"))
    )
    rpt.para(
        f"Temurtas et al. {c('temurtas2009')} put an LM-trained multilayer net next to a probabilistic neural "
        "network, with both a split and 10-fold CV. The multilayer net hit 82.37%. The PNN hit 78.13%. Later "
        f"tables copy those numbers {c('erkaymaz2017')}. Erkaymaz et al. {c('erkaymaz2017')} rewired a "
        "feed-forward net into small-world graphs. Newman–Watts: 93.06% accuracy, sensitivity 0.85, "
        "specificity 0.96. Watts–Strogatz: 91.66%. Ordinary feed-forward: 83.33%. The graphs get fancier. "
        "Accuracy is still the number most papers lead with."
    )
    rpt.section("2.3 Classical and Ensemble Machine-Learning Approaches")
    rpt.para(
        f"Classical models sit in a narrower band. Maniruzzaman et al. {c('maniruzzaman2017')} ran "
        "Gaussian-process classification against LDA, QDA and naïve Bayes under several CV setups. The "
        "Gaussian-process model quoted 81.97% accuracy, 91.79% sensitivity, 63.33% specificity. High recall, "
        f"weak specificity. Sisodia and Sisodia {c('sisodia2018')} trained a tree, an SVM and naïve Bayes and "
        "actually reported precision, recall, F-measure and ROC. Naïve Bayes won on accuracy at 76.30%. Zou "
        f"et al. {c('zou2018')} tried trees, forests and nets with PCA and mRMR, both on a Luzhou hospital "
        f"set and on Pima. Joshi and Dhakal {c('joshi2021')} kept it simpler: logistic regression and a "
        "decision tree. Logistic hit 78.26% and the useful predictors were pregnancies, glucose, BMI, "
        f"pedigree and age. Chang et al. {c('chang2023')} compared naïve Bayes, a random forest and J48 on a "
        "70/30 split. Forest: 79.57% accuracy, AUC 86.24% on the full feature set."
    )
    rpt.para(
        f"The headline numbers jump once people stack ensembles and heavy preprocessing. Hasan et al. "
        f"{c('hasan2020')} piled on outlier rejection, filling, standardisation, feature selection and k-fold "
        "CV, then several classifiers including an MLP. Weighted ensemble: AUC 0.950, sensitivity 0.789, "
        f"specificity 0.934. Kumari et al. {c('kumari2021')} stayed milder. Soft vote of random forest, "
        "logistic regression and naïve Bayes: 79.04%. Howlader et al. {c('howlader2022')} tried a lot of "
        "selectors and classifiers and quoted 90.91% for generalised boosted regression. That is well above "
        f"the usual 75–82% band. Kavakiotis et al. {c('kavakiotis2017')} survey the wider diabetes ML "
        "literature if you want the bigger picture."
    )
    rpt.section("2.4 Methodological Literature")
    rpt.para(
        "A few methods papers are the actual bar we tried to meet. Kohavi argued for stratified 10-fold CV "
        f"when you pick a model and when you quote its accuracy {c('kohavi1995')}. Varma and Simon "
        f"{c('varma2006')} and Cawley and Talbot {c('cawley2010')} showed the trap: quote the CV score of a "
        "setup that was itself chosen by CV, and the number is biased high. Nested CV takes that bump out. "
        "Kaufman et al. {0} spelled out what counts as leakage. Kapoor and Narayanan {1} collected cases "
        "where it wrecked reproducibility. scikit-learn's own docs say every data-dependent step should sit "
        "inside a pipeline that is refitted in each fold {2}. We treated that as a rule, not a suggestion."
        .format(c("kaufman2012"), c("kapoor2023"), c("sklearn_pitfalls"))
    )
    rpt.para(
        f"Imbalance is its own mess. He and Garcia {c('he2009')} cover resampling and cost-sensitive tricks. "
        f"Chawla et al. {c('chawla2002')} introduced SMOTE, which interpolates extra minority rows. Fine, as "
        "long as it stays on the training side. Copies or interpolations of a test patient leaking into "
        f"training is the usual failure. Fawcett {c('fawcett2006')} is still the ROC walkthrough we used. "
        f"Davis and Goadrich {c('davis2006')} relate ROC to precision–recall. Saito and Rehmsmeier "
        f"{c('saito2015')} make the point that PR plots say more than ROC when positives are the minority. "
        f"Little and Rubin {c('little2019')} say when a simple imputer is acceptable. That stack is the "
        "evaluation standard we held ourselves to."
    )
    rpt.section("2.5 Comparison of Existing Work")
    rpt.para(
        f"{rpt.next_table_label()} lines the surveyed papers up by method, evaluation protocol, reported "
        "result, and the limitation that matters for this project. \"Accuracy\" here means the figure the "
        "authors quoted on their own protocol."
    )
    lit_rows = [
        ["Ref.", "Technique", "Evaluation", "Reported result", "Limitation / observation"],
        [c("smith1988"), "ADAP (early neural model)", "576/192 split", "Sens. = Spec. = 76%; ROC",
         "Single split; early algorithm"],
        [c("kayaer2003"), "GRNN, RBF, MLP-LM", "Train/test split", "GRNN 80.21%; MLP-LM 77.08%",
         "Accuracy only; single split"],
        [c("polat2007"), "PCA (8→4) + ANFIS", "10-fold CV", "Accuracy 89.47%",
         "Accuracy only"],
        [c("kahramanli2008"), "Hybrid ANN + fuzzy NN", "k-fold CV", "Accuracy 84.24%",
         "Complex hybrid; no classical baseline"],
        [c("temurtas2009"), "MLNN-LM, PNN", "Split and 10-fold CV", "MLNN 82.37%; PNN 78.13%",
         "Accuracy only"],
        [c("bozkurt2014"), "Comparison of earlier classifiers", "Literature + experiments",
         "Most earlier studies report accuracy only", "Highlights missing sensitivity/specificity"],
        [c("erkaymaz2017"), "Small-world feed-forward ANN", "k-fold CV", "93.06%; Sens. 0.85, Spec. 0.96",
         "Specialised topology; not independently replicated"],
        [c("maniruzzaman2017"), "Gaussian process, LDA, QDA, NB", "K2–K10 CV", "GPC 81.97%; Sens. 91.79%; Spec. 63.33%",
         "Low specificity at reported operating point"],
        [c("sisodia2018"), "DT, SVM, naïve Bayes", "Cross-validation", "NB accuracy 76.30%",
         "No neural network compared"],
        [c("hasan2020"), "Outlier removal, selection, MLP & ensembles", "k-fold CV",
         "AUC 0.950; Sens. 0.789; Spec. 0.934", "Outlier rejection changes the evaluated population"],
        [c("joshi2021"), "Logistic regression, decision tree", "Train/test + CV", "Accuracy 78.26%",
         "No neural network compared"],
        [c("kumari2021"), "Soft-voting RF + LR + NB", "Train/test", "Accuracy 79.04%", "No neural network in ensemble"],
        [c("howlader2022"), "Many selectors and classifiers", "Cross-validation", "GBR accuracy 90.91%",
         "Well above the typical 75–82% range"],
        [c("chang2023"), "NB, RF, J48", "70/30 split", "RF 79.57%; AUC 86.24%", "Single split"],
        ["This work", "Leakage-safe MLP 8–16–8–1 + training-fold ROS", "10×5 repeated CV + hold-out",
         f"ROC-AUC {rep(PROPOSED, 'roc_auc'):.3f}; Sens. {rep(PROPOSED, 'recall_sensitivity'):.3f}; "
         f"Spec. {rep(PROPOSED, 'specificity'):.3f}", "Single population; no external cohort"],
    ]
    rpt.table(lit_rows, "Comparison of published work on the Pima Indians Diabetes dataset",
              widths=[0.5, 1.45, 1.05, 1.55, 1.45], size=10, align=["c", "l", "l", "l", "l"],
              bold_rows=(len(lit_rows) - 1,), short="Comparison of published work on the Pima dataset")
    rpt.section("2.6 Research Gap")
    rpt.para(
        "Nobody needed another reminder that an ANN can be trained on Pima. Smith et al. did that in 1988. "
        "Dozens of papers followed. The hole is in how the work is done. You can see it in "
        f"{rpt.tables[-1][0]}:"
    )
    rpt.bullets([
        "Papers treat the zero codes differently — keeping them, recoding every zero including valid zero "
        "pregnancies, or dropping rows — and often do not say which rule they used.",
        "If preprocessing, feature selection, oversampling, or hyperparameter search is fitted on the full "
        "table or tuned on test data, the quoted scores look better than they are. Many write-ups do not make "
        "the protocol clear.",
        "Accuracy is still the headline number in most studies, even though the majority-class floor is 65.1%. "
        "None of the surveyed papers reports PR-AUC.",
        "ANNs are often shown without same-fold baselines, repeated cross-validation, or a significance test, "
        "so it is hard to tell whether the network actually beats logistic regression.",
        "How much each protocol shortcut inflates the scores is rarely measured on this dataset.",
    ])
    rpt.para(
        "We tried to close that gap with a leakage-safe ANN pipeline, same-fold baselines plus corrected "
        "tests, a full metric set, an ablation, and a leakage run that actually measures how much the usual "
        "shortcuts puff the scores."
    )

    # ============================================================ CHAPTER 3
    rpt.chapter_heading("Proposed Methodology")
    rpt.section("3.1 System Overview and Block Schematic")
    fig_block = rpt.next_figure_label()
    rpt.para(
        f"{fig_block} is the whole pipeline on one page. Stage A just loads the local file, looks at it, and "
        "cuts it into stratified folds. Stage B is the training side. For each training fold we recode the "
        "impossible zeros, learn the medians, maybe pick features, learn the scaler, oversample the minority "
        "class, and fit the ANN. Stage C is the test fold. It goes through those already-fitted objects. No "
        "refitting. No extra resampling. Then the trained net, the 0.5 cut, and the scores. Stage D is "
        "bookkeeping: fold scores, baseline comparisons, ablation, leakage runs, hold-out."
    )
    rpt.figure(FIG / "fig_block_schematic.png",
               "Block schematic of the proposed leakage-safe ANN-based Diabetes Prediction System. Learned steps "
               "(Stage B) are fitted on the training fold only; the test fold (Stage C) is transformed with the "
               "fitted objects and scored once.", width=6.0, short="Block schematic of the proposed system")
    rpt.section("3.2 Problem Formulation")
    rpt.para(
        "Write D = {(*x*_{i}, *y*_{i})}, *i* = 1, …, *N* with *N* = 768. Each *x*_{i} is the eight "
        "measurements. Each *y*_{i} is 0 or 1. We learn a function *f*_{θ} that estimates the chance of a "
        "positive label. A threshold τ then turns that chance into a class:"
    )
    rpt.equation(r"\hat{p}_i = f_{\theta}(x_i) \approx P(y_i = 1 \mid x_i), \qquad f_{\theta}:\mathbb{R}^{8}\rightarrow[0,1]")
    rpt.equation(r"\hat{y}_i = \mathbf{1}\left[\hat{p}_i \geq \tau\right], \qquad \tau = 0.5")
    rpt.para(
        "Scores are always computed on rows that were not used to fit any part of the pipeline. If the indices "
        "are split into a training part *T* and a test part *S*, then medians, means, variances, F-scores, "
        "resampled rows, and network weights are all functions of the rows in *T* only."
    )
    rpt.section("3.3 Data Preprocessing")
    rpt.lead("Zero recoding.",
             "Five measurements cannot be zero in a living adult: plasma glucose, diastolic blood pressure, "
             "triceps skinfold, 2-hour serum insulin, and BMI. Those zeros are turned into missing values "
             "(NaN) by a fixed rule that does not use any statistics, so it can be applied the same way to "
             "every row. Zero pregnancies is a real value and is kept. With Z the set of recoded columns:")
    e_zero = rpt.equation(r"\tilde{x}_{ij} = \mathrm{NaN}\ \ \mathrm{if}\ \ j\in Z\ \mathrm{and}\ x_{ij}=0;\quad \tilde{x}_{ij}=x_{ij}\ \ \mathrm{otherwise},\quad Z=\{\mathrm{Glu,BP,Skin,Ins,BMI}\}")
    rpt.lead("Median imputation.",
             "Each missing value in column *j* is filled with the median of the observed training-fold values "
             f"of that column {c('little2019')}. The median holds up against the long right tails of insulin "
             "and the pedigree function:")
    e_med = rpt.equation(r"m_j = \mathrm{median}\{\tilde{x}_{ij} : i \in T,\ \tilde{x}_{ij} \neq \mathrm{NaN}\}, \qquad \hat{x}_{ij} = m_j\ \ \mathrm{if}\ \tilde{x}_{ij} = \mathrm{NaN}")
    rpt.lead("Univariate feature selection (optional).",
             "When it is switched on, SelectKBest keeps the *k* features with the largest one-way ANOVA F "
             f"statistic on the training fold {c('guyon2003')}. For feature *j* with class means "
             "x̄_{jc}, overall mean x̄_{j}, *n*_{c} rows in class *c*, and *C* = 2 classes:")
    e_f = rpt.equation(r"F_j = \dfrac{\sum_{c} n_c\,(\bar{x}_{jc}-\bar{x}_{j})^{2}/(C-1)}{\sum_{c}\sum_{i\in c}(x_{ij}-\bar{x}_{jc})^{2}/(n-C)}")
    rpt.lead("Standardisation.",
             "Each feature is centred and scaled with the mean μ_{j} and standard deviation σ_{j} of the "
             "training fold. That puts glucose in mg/dL and the pedigree score on similar scales for "
             "gradient-based training:")
    e_z = rpt.equation(r"z_{ij} = \dfrac{\hat{x}_{ij} - \mu_j}{\sigma_j}, \qquad \mu_j = \dfrac{1}{|T|}\sum_{i\in T}\hat{x}_{ij}, \quad \sigma_j^{2} = \dfrac{1}{|T|}\sum_{i\in T}(\hat{x}_{ij}-\mu_j)^{2}")
    rpt.lead("Random oversampling of the training fold.",
             f"To counter the 65:35 imbalance {c('he2009')}, minority-class rows of the training fold are "
             "copied at random (with replacement) until both classes have the same count. If *I*_{1} is the "
             "index set of positive training rows and *n*_{0}, *n*_{1} are the class counts, the resampled "
             "training set is:")
    e_ros = rpt.equation(r"T' = T \cup \{i_1, \ldots, i_{n_0-n_1}\}, \qquad i_k \sim \mathrm{Uniform}(I_1)")
    rpt.para(
        f"Oversampling is done with imbalanced-learn {c('lemaitre2017')} inside the pipeline, so it runs only "
        "during fitting. The test fold is never resampled. That keeps its class balance the same as the "
        f"population and stops copies of a test patient from entering training. SMOTE {c('chawla2002')} is "
        "tried as an alternative in the ablation study."
    )
    rpt.section("3.4 Artificial Neural Network Architecture")
    fig_ann = rpt.next_figure_label()
    rpt.para(
        f"The classifier is a fully connected feed-forward network with eight inputs, two hidden layers of 16 "
        f"and 8 units, and one output ({fig_ann}). Hidden units use ReLU {c('nair2010')}; the output uses a "
        "logistic sigmoid so it can be read as a probability. The network has "
        "(8·16 + 16) + (16·8 + 8) + (8·1 + 1) = 289 trainable parameters. That is small next to the roughly "
        "800 rows of an oversampled training fold, which helps keep over-fitting in check."
    )
    rpt.figure(FIG / "fig_ann_architecture.png",
               "Architecture of the proposed feed-forward ANN (8–16–8–1) with ReLU hidden layers and a sigmoid "
               "output.", width=5.6, short="Architecture of the proposed ANN (8–16–8–1)")
    rpt.lead("Forward propagation.", "With *a*^{(0)} = *z* the standardised input, each hidden layer does an "
             "affine transform then ReLU, and the output layer applies the sigmoid:")
    e_hid = rpt.equation(r"h^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \mathrm{ReLU}(h^{(l)}) = \max(0, h^{(l)}), \qquad l = 1, 2")
    e_out = rpt.equation(r"\hat{p} = \sigma\left(W^{(3)}a^{(2)} + b^{(3)}\right), \qquad \sigma(u) = \dfrac{1}{1 + e^{-u}}")
    rpt.para("where *W*^{(1)}, *W*^{(2)}, and *W*^{(3)} are matrices of size 16 × 8, 8 × 16, and 1 × 8 with matching bias vectors. "
             f"Weights are initialised with the Glorot (Xavier) uniform scheme {c('glorot2010')}, where "
             "*n*_{in} and *n*_{out} are the fan-in and fan-out of the layer:")
    e_init = rpt.equation(r"W^{(l)}_{rs} \sim \mathcal{U}\left(-\sqrt{\dfrac{6}{n_{in}+n_{out}}},\ \sqrt{\dfrac{6}{n_{in}+n_{out}}}\right)")
    rpt.lead("Loss function.", "The network minimises mean binary cross-entropy over a mini-batch *B*, plus "
             "an L2 penalty on the weights with strength α = 10^{−3} (weight decay):")
    e_loss = rpt.equation(r"\mathcal{L}(\theta) = -\dfrac{1}{|B|}\sum_{i\in B}\left[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] + \dfrac{\alpha}{2|B|}\sum_{l=1}^{3}\|W^{(l)}\|_{F}^{2}")
    rpt.lead("Back-propagation.", f"Gradients come from the chain rule {c('rumelhart1986', 'goodfellow2016')}. "
             "For a sigmoid output with cross-entropy loss the output error is just the prediction minus the "
             "label. Errors are sent backwards through the ReLU derivative, and the weight gradient is the "
             "outer product of the error with the previous layer's activation:")
    e_delta = rpt.equation(r"\delta^{(3)} = \hat{p} - y, \qquad \delta^{(l)} = \left(W^{(l+1)\top}\delta^{(l+1)}\right)\odot\mathbf{1}\left[h^{(l)} > 0\right],\quad l = 2, 1")
    e_grad = rpt.equation(r"\dfrac{\partial\mathcal{L}}{\partial W^{(l)}} = \dfrac{1}{|B|}\sum_{i\in B}\delta_i^{(l)}\,a_i^{(l-1)\top} + \dfrac{\alpha}{|B|}W^{(l)}, \qquad \dfrac{\partial\mathcal{L}}{\partial b^{(l)}} = \dfrac{1}{|B|}\sum_{i\in B}\delta_i^{(l)}")
    rpt.lead("Adam optimiser.", f"Parameters are updated with Adam {c('kingma2015')} on mini-batches of 32 "
             "rows, with learning rate η = 10^{−3}, β_{1} = 0.9, β_{2} = 0.999, and ε = 10^{−8}. With gradient "
             "*g*_{t} at step *t*:")
    e_adam1 = rpt.equation(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^{2}")
    e_adam2 = rpt.equation(r"\hat{m}_t = \dfrac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \dfrac{v_t}{1-\beta_2^{t}}, \qquad \theta_t = \theta_{t-1} - \eta\,\dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}")
    rpt.lead("Early stopping.", f"A stratified 15% of the (resampled) training fold is held out as a "
             f"validation set {c('prechelt1998')}. Training stops when validation accuracy has not improved by "
             "at least 10^{−4} for 20 epochs in a row (maximum 500), and the weights from the best validation "
             "epoch are restored. Early stopping and the L2 penalty are the two regularisers of the network.")
    rpt.section("3.5 Baseline Models")
    rpt.para(
        "Three baselines go through the same pipeline and the same folds. The **dummy classifier** always "
        "predicts the majority class and sets the floor (accuracy ≈ 0.651, sensitivity 0, ROC-AUC 0.5). "
        f"**Logistic regression** {c('cox1958')} is a linear model of the log-odds with L2 regularisation "
        "(C = 1) and balanced class weights, solved with liblinear:"
    )
    rpt.equation(r"\hat{p} = \sigma(w^{\top}z + b), \qquad \omega_c = \dfrac{N}{2N_c}\ \ \mathrm{(balanced\ class\ weight\ for\ class}\ c)")
    rpt.para(
        f"The **random forest** {c('breiman2001')} averages 100 decision trees grown on bootstrap samples with a "
        "random subset of √8 ≈ 3 features considered at each split, maximum depth 6, at least 5 rows per leaf, "
        "and balanced class weights. Splits minimise Gini impurity, and the forest's probability is the mean "
        "of the trees' leaf class fractions:"
    )
    rpt.equation(r"G(t) = 1 - \sum_{c\in\{0,1\}} p_{c}(t)^{2}, \qquad \hat{p}_{RF}(x) = \dfrac{1}{B}\sum_{b=1}^{B}\hat{p}_{b}(x),\quad B = 100")
    rpt.section("3.6 Evaluation Protocol and Metrics")
    fig_cv = rpt.next_figure_label()
    rpt.para(
        f"Stratified *k*-fold cross-validation with *k* = 5 {c('kohavi1995')} splits the data into five folds "
        f"that keep the same class ratio ({fig_cv}). Each model is trained five times, each time on four "
        "folds, and scored on the remaining fold. Every model sees the same splits. For significance testing "
        "the procedure is repeated ten times with different shuffles (50 splits). Separately, a single "
        "stratified 80/20 split gives a hold-out test set that is scored once after all design decisions were "
        "fixed."
    )
    rpt.figure(FIG / "fig_cv_protocol.png", "Stratified 5-fold cross-validation with the full pipeline refitted "
               "inside each training fold.", width=5.8, short="Stratified 5-fold cross-validation protocol")
    rpt.para(
        "Write TP, TN, FP, and FN for the true positives, true negatives, false positives, and false negatives "
        "at threshold τ. The threshold-dependent metrics are then:"
    )
    e_acc = rpt.equation(r"\mathrm{Accuracy} = \dfrac{TP+TN}{TP+TN+FP+FN}, \qquad \mathrm{Precision} = \dfrac{TP}{TP+FP}")
    rpt.equation(r"\mathrm{Sensitivity} = \mathrm{Recall} = \dfrac{TP}{TP+FN}, \qquad \mathrm{Specificity} = \dfrac{TN}{TN+FP}")
    e_f1 = rpt.equation(r"F_1 = \dfrac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}")
    rpt.para(
        f"Two threshold-free metrics summarise ranking quality. ROC-AUC {c('fawcett2006', 'hanley1982')} is "
        "the chance that a randomly chosen positive is scored above a randomly chosen negative. PR-AUC is "
        f"computed as average precision {c('davis2006', 'saito2015')}. It puts more weight on the positive class "
        "and has a baseline equal to the positive rate (0.349) rather than 0.5:"
    )
    e_auc = rpt.equation(r"\mathrm{AUC} = \dfrac{1}{n_{+}n_{-}}\sum_{i:y_i=1}\ \sum_{j:y_j=0}\mathbf{1}[\hat{p}_i > \hat{p}_j], \qquad \mathrm{AP} = \sum_{n}(R_n - R_{n-1})\,P_n")
    rpt.para(
        f"The operating point is looked at with Youden's index {c('youden1950')}, which is largest at the "
        "threshold that best balances sensitivity and specificity:"
    )
    rpt.equation(r"J(\tau) = \mathrm{Sensitivity}(\tau) + \mathrm{Specificity}(\tau) - 1")
    rpt.para(
        "The 50 repeated-CV scores share training rows, so a naive paired t-test underestimates the variance. "
        f"The corrected resampled t-test of Nadeau and Bengio {c('nadeau2003')} inflates the variance of the "
        "per-split differences *d*_{j} by the test-to-training ratio:"
    )
    rpt.equation(r"t = \dfrac{\bar{d}}{\sqrt{\left(\dfrac{1}{J} + \dfrac{n_{test}}{n_{train}}\right)s_d^{2}}}, \qquad J = 50,\ \ \dfrac{n_{test}}{n_{train}} \approx \dfrac{154}{614}")
    rpt.para(
        f"Hold-out metrics are given with 95% percentile bootstrap intervals from 2,000 resamples of the test "
        f"set {c('efron1993')}. Feature importance is measured by permutation {c('breiman2001')}: the drop in "
        "hold-out ROC-AUC when one feature column is randomly shuffled, averaged over *K* = 30 repeats:"
    )
    rpt.equation(r"I_j = s(f, S) - \dfrac{1}{K}\sum_{k=1}^{K} s(f, S_{k,j}^{\pi})")
    rpt.section("3.7 Algorithms")
    rpt.para("Algorithm 1 gives the leakage-safe training and evaluation procedure, Algorithm 2 the training "
             "loop of the ANN, and Algorithm 3 the inference step for a new record.")
    def eq(*labels: str) -> str:
        nums = [lab.strip("()") for lab in labels]
        return f"Eq. {nums[0]}" if len(nums) == 1 else f"Eqs. {nums[0]}–{nums[-1]}"

    rpt.algorithm("Leakage-safe stratified cross-validation of a pipeline", [
        (0, "**Input:** data D = {(x_{i}, y_{i})}, i = 1..N; classifier g; folds k = 5; seed s = 42"),
        (0, "Z ← {Glucose, BloodPressure, SkinThickness, Insulin, BMI}"),
        (0, "Split indices into k stratified folds S_{1}, …, S_{k} using seed s"),
        (0, "**for** f = 1 **to** k **do**"),
        (1, "T ← {1..N} \\ S_{f}   // training indices; S_{f} is never used for fitting"),
        (1, f"Recode zeros in columns Z of rows T and S_{{f}} as missing (fixed rule, {eq(e_zero)})"),
        (1, f"m ← column medians of rows T; fill missing values of T and S_{{f}} with m ({eq(e_med)})"),
        (1, f"**if** selection enabled **then** keep the k columns with the largest F-scores on T ({eq(e_f)})"),
        (1, f"μ, σ ← mean and std of rows T; standardise T and S_{{f}} with μ, σ ({eq(e_z)})"),
        (1, f"T′ ← random oversampling of the minority class in T ({eq(e_ros)})"),
        (1, "Train g on T′ (Algorithm 2 for the ANN)"),
        (1, "p̂_{i} ← g(z_{i}) and ŷ_{i} ← 1[p̂_{i} ≥ 0.5] for every row i of S_{f}"),
        (1, f"Store fold metrics ({eq(e_acc, e_auc)}) and out-of-fold p̂_{{i}}"),
        (0, "**end for**"),
        (0, "**return** mean ± std of fold metrics, out-of-fold confusion matrix, ROC and PR curves"),
    ])
    rpt.algorithm("Training the ANN with Adam and early stopping", [
        (0, "**Input:** standardised, resampled training set T′; α = 10^{−3}; η = 10^{−3}; batch 32; patience 20"),
        (0, "Split T′ into training part T_{a} (85%) and validation part V (15%), stratified"),
        (0, f"Initialise W^{{(l)}} by Glorot uniform ({eq(e_init)}), b^{{(l)}} ← 0; best ← −∞; wait ← 0"),
        (0, "**for** epoch = 1 **to** 500 **do**"),
        (1, "Shuffle T_{a} and partition it into mini-batches B of 32 rows"),
        (1, "**for each** mini-batch B **do**"),
        (2, f"Forward pass: h^{{(l)}}, a^{{(l)}}, p̂ ({eq(e_hid, e_out)}); loss L(θ) ({eq(e_loss)})"),
        (2, f"Backward pass: δ^{{(3)}}, δ^{{(2)}}, δ^{{(1)}} and gradients ({eq(e_delta, e_grad)})"),
        (2, f"Adam update of every W^{{(l)}}, b^{{(l)}} ({eq(e_adam1, e_adam2)})"),
        (1, "**end for**"),
        (1, "acc ← accuracy of the network on V"),
        (1, "**if** acc > best + 10^{−4} **then** best ← acc; θ_{best} ← θ; wait ← 0 **else** wait ← wait + 1"),
        (1, "**if** wait = 20 **then break**"),
        (0, "**end for**"),
        (0, "**return** θ_{best} (weights of the best validation epoch)"),
    ])
    rpt.algorithm("Inference for a new record", [
        (0, "**Input:** raw record x of 8 values; fitted objects (Z, m, μ, σ, θ_{best})"),
        (0, "**for each** column j in Z **do if** x_{j} = 0 **then** x_{j} ← m_{j}   // impossible zero treated as missing"),
        (0, "**for** j = 1 **to** 8 **do if** x_{j} is missing **then** x_{j} ← m_{j}"),
        (0, "z ← (x − μ) / σ"),
        (0, "p̂ ← σ(W^{(3)} ReLU(W^{(2)} ReLU(W^{(1)} z + b^{(1)}) + b^{(2)}) + b^{(3)})"),
        (0, "**return** p̂ and ŷ = 1[p̂ ≥ 0.5]   // educational risk estimate, not a diagnosis"),
    ])
    rpt.section("3.8 Implementation Details")
    rpt.para(
        f"We wrote it in Python 3.9. scikit-learn 1.5.2 {c('pedregosa2011')}, imbalanced-learn "
        f"{c('lemaitre2017')}, NumPy 2.0 {c('harris2020')}, pandas 2.3 {c('mckinney2010')}, Matplotlib 3.9 "
        f"{c('hunter2007')}. Small package: load, profile, preprocess, train, score, plot. Twenty unit tests. "
        "The ones we actually cared about: Pregnancies = 0 is never recoded, and imputation statistics come "
        "from the training fold only. Seed 42 everywhere. Table 3.1 lists the hyperparameters. None of them "
        "was tuned on a test fold."
    )
    rpt.table([
        ["Component", "Setting"],
        ["Zero recoding", "Glucose, BloodPressure, SkinThickness, Insulin, BMI: 0 → missing; Pregnancies kept"],
        ["Imputation / scaling", "Median imputation; z-score standardisation (training-fold statistics)"],
        ["Oversampling", "RandomOverSampler to a 1:1 class ratio, training fold only"],
        ["ANN architecture", "8 → 16 (ReLU) → 8 (ReLU) → 1 (sigmoid); 289 parameters"],
        ["ANN training", "Adam, η = 0.001, batch 32, α = 0.001, ≤ 500 epochs, early stopping (15%, patience 20)"],
        ["Logistic regression", "L2, C = 1, liblinear, balanced class weights"],
        ["Random forest", "100 trees, max depth 6, min. 5 rows per leaf, balanced class weights"],
        ["Feature selection", "SelectKBest, ANOVA F, k = 5 (optional; ablation k = 3–8)"],
        ["Validation", "Stratified 5-fold (seed 42); 10 × 5 repeated CV; 80/20 hold-out"],
    ], "Hyperparameters and settings of the proposed system and baselines", widths=[1.6, 4.4],
        short="Hyperparameters and settings")

    # ============================================================ CHAPTER 4
    rpt.chapter_heading("Experimental Results and Discussions")
    rpt.section("4.1 Experimental Setup")
    rpt.para(
        "Everything was run on an Apple M5 laptop CPU, with no GPU. The full suite — 13 seeded "
        "cross-validation configurations, 250 repeated-CV fits, 35 ablation settings with 15 fits each, plus "
        "the leakage, nested-CV, seed-stability, and hold-out runs — took about "
        f"{RES['runtime_seconds'] / 60:.0f} minutes. Four evaluation protocols are used, each for a different "
        "job:"
    )
    rpt.bullets([
        "**P1 — seeded 5-fold CV (seed 42):** the comparison of 13 configurations, out-of-fold ROC/PR curves, "
        "and confusion matrices.",
        "**P2 — 10 × 5 repeated stratified CV (50 splits):** headline estimates and corrected t-tests for the "
        "five main models.",
        "**P3 — 3 × 5 repeated stratified CV (15 splits):** the ablation and leakage experiments, with the same "
        "splits for every configuration.",
        "**P4 — stratified 80/20 hold-out (614/154 rows):** a single final test after all choices were fixed.",
    ])
    rpt.section("4.2 Dataset Details")
    rpt.para(
        f"{rpt.next_table_label()} summarises the dataset and {rpt.next_table_label(2)} describes each feature "
        "from the local file. There are 768 rows and 9 columns, with no duplicate rows and no empty cells. "
        "The only missingness is in the form of zero codes."
    )
    rpt.table([
        ["Property", "Value"],
        ["Name", "Pima Indians Diabetes Database (UCI extract, 1990)"],
        ["Original owner", "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)"],
        ["Population", "Women aged ≥ 21 years of Pima heritage, near Phoenix, Arizona, USA"],
        ["Instances / predictors", "768 / 8 numeric predictors + binary Outcome"],
        ["Class distribution", "Outcome 0: 500 (65.1%); Outcome 1: 268 (34.9%)"],
        ["Outcome definition", "2-hour post-load plasma glucose ≥ 200 mg/dL (WHO criterion)"],
        ["Duplicates / empty cells", "0 / 0 (missing values are coded as 0)"],
        ["Majority-class accuracy", "0.651 (floor any useful model must beat)"],
    ], "Summary of the Pima Indians Diabetes dataset", widths=[1.8, 4.2], short="Dataset summary")
    med = EDA["class_medians_recoded"]
    rc = DF.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        rc[col] = rc[col].mask(rc[col] == 0)
    desc = {
        "Pregnancies": "Number of pregnancies",
        "Glucose": "2-h OGTT plasma glucose (mg/dL)",
        "BloodPressure": "Diastolic blood pressure (mm Hg)",
        "SkinThickness": "Triceps skinfold (mm)",
        "Insulin": "2-h serum insulin (µU/mL)",
        "BMI": "Body-mass index (kg/m²)",
        "DiabetesPedigreeFunction": "Family-history score",
        "Age": "Age (years)",
    }
    rows = [["Feature", "Description (unit)", "Range (raw)", "Zeros", "Missing after recode",
             "Median y = 0 / y = 1", "r with y"]]
    for col in FEATURE_COLUMNS:
        lo, hi = DF[col].min(), DF[col].max()
        fmt_v = (lambda v: f"{v:.3f}") if col == "DiabetesPedigreeFunction" else (
            (lambda v: f"{v:.1f}") if col == "BMI" else (lambda v: f"{v:.0f}"))
        miss = int(rc[col].isna().sum())
        rows.append([
            SHORT[col], desc[col], f"{fmt_v(lo)}–{fmt_v(hi)}", str(int(zeros[col])),
            f"{miss} ({miss / len(DF):.1%})" if col in ZERO_AS_MISSING_COLUMNS else "0 (kept)",
            f"{fmt_v(med[col]['0'])} / {fmt_v(med[col]['1'])}", f"{EDA['corr_with_outcome'][col]:.3f}",
        ])
    rpt.table(rows, "Feature dictionary and statistics (medians and correlations after recoding zeros)",
              widths=[1.05, 1.45, 0.75, 0.5, 0.85, 0.85, 0.55], size=10,
              align=["l", "l", "c", "c", "c", "c", "c"], short="Feature dictionary and statistics")
    fig_bal = rpt.next_figure_label()
    fig_zero = rpt.next_figure_label(2)
    rpt.para(
        f"{fig_bal} shows the class balance. Because 65.1% of rows are negative, accuracy on its own cannot "
        f"tell a useful model from one that simply ignores the minority class. {fig_zero} shows the zeros. "
        f"Insulin has {zeros['Insulin']} of them (48.7% of rows) and SkinThickness {zeros['SkinThickness']} "
        f"(29.6%). BloodPressure ({zeros['BloodPressure']}), BMI ({zeros['BMI']}), and Glucose "
        f"({zeros['Glucose']}) have fewer. Those zeros are impossible in a living adult, so they are treated "
        f"as missing. The {zeros['Pregnancies']} zeros in Pregnancies are real (never pregnant) and are kept. "
        "Recoding them would throw away information; Chapter 5 checks that this choice is right."
    )
    rpt.figure(FIG / "fig_class_balance.png", "Outcome class balance of the 768 rows.", width=4.2,
               short="Outcome class balance")
    rpt.figure(FIG / "fig_zero_counts.png", "Zero values per feature in the raw file; orange bars are recoded "
               "as missing, blue bars are kept.", width=5.2, short="Zero values per feature")
    rpt.section("4.3 Exploratory Data Analysis")
    fig_dist = rpt.next_figure_label()
    fig_box = rpt.next_figure_label(2)
    fig_corr = rpt.next_figure_label(3)
    cm_ = EDA["class_means_recoded"]
    rpt.para(
        f"{fig_dist} compares each feature in the two classes after recoding. Glucose is the clearest split: "
        f"the median is {med['Glucose']['0']:.0f} mg/dL for negative rows and {med['Glucose']['1']:.0f} mg/dL "
        f"for positive ones. Positive rows also have higher BMI (median {med['BMI']['0']:.1f} vs "
        f"{med['BMI']['1']:.1f}), are older (median {med['Age']['0']:.0f} vs {med['Age']['1']:.0f} years), and "
        "have more pregnancies. Insulin and skinfold separate the classes once zeros are removed (mean insulin "
        f"{cm_['Insulin']['0']:.0f} vs {cm_['Insulin']['1']:.0f} µU/mL); in the raw file the zeros hide that. "
        "Blood pressure and the pedigree function overlap a lot between the two classes."
    )
    rpt.figure(FIG / "fig_feature_distributions.png", "Distribution of each feature by outcome after recoding "
               "impossible zeros; dashed lines mark class medians.", short="Feature distributions by outcome")
    rpt.para(
        f"The box plots in {fig_box} show long right tails for insulin, the pedigree function, and age, plus "
        "a few extreme BMI and skinfold values. The raw 1.5 × IQR counts are modest — 34 for insulin and 29 "
        "for the pedigree function, for example. Those points were looked at but not deleted. Dropping them "
        "on the full table would use test-fold information and shrink an already small dataset, and the median "
        "imputer and the small regularised network are not very sensitive to a few extremes."
    )
    rpt.figure(FIG / "fig_boxplots.png", "Box plots of each feature by outcome after recoding (points beyond "
               "1.5 × IQR shown).", short="Box plots by outcome")
    corr = EDA["corr_with_outcome"]
    rpt.para(
        f"{fig_corr} shows Pearson correlations after recoding. Glucose is most associated with the outcome "
        f"(r = {corr['Glucose']:.3f}), then BMI ({corr['BMI']:.3f}), insulin ({corr['Insulin']:.3f}), "
        f"skinfold ({corr['SkinThickness']:.3f}), and age ({corr['Age']:.3f}). Among the predictors, skinfold "
        "and BMI (r ≈ 0.65), glucose and insulin (r ≈ 0.58), and age and pregnancies (r ≈ 0.54) are the most "
        "redundant pairs. None of those pairs is so strong that a feature has to be dropped, and no single "
        "feature explains the outcome on its own, so a multivariate model is still needed."
    )
    rpt.figure(FIG / "fig_correlation.png", "Pearson correlation matrix after recoding zeros (pairwise "
               "complete observations).", width=4.8, short="Correlation matrix")
    rpt.section("4.4 Feature Relevance and Selection")
    fig_fs = rpt.next_figure_label()
    an = EDA["anova_train_folds"]
    rpt.para(
        f"{fig_fs} (left) shows ANOVA F-scores computed separately on each training fold. Glucose dominates "
        f"(F ≈ {an['Glucose'][0]:.0f}), then BMI (≈ {an['BMI'][0]:.0f}), age (≈ {an['Age'][0]:.0f}), and "
        f"pregnancies (≈ {an['Pregnancies'][0]:.0f}). The right panel shows how often SelectKBest with k = 5 "
        "kept each feature: glucose, BMI, and age in all five folds, skinfold in four, pregnancies and insulin "
        "in three, and blood pressure and the pedigree function never. The selected set changes from fold to "
        "fold, which is what should happen when selection is refitted on each training fold, and it is a check "
        "that the set was not fixed in advance on all rows."
    )
    rpt.figure(FIG / "fig_feature_selection.png", "ANOVA F-scores on each training fold (left) and SelectKBest "
               "(k = 5) selection frequency across the five folds (right).", short="ANOVA F-scores and selection frequency")
    rpt.section("4.5 Training Behaviour of the ANN")
    fig_loss = rpt.next_figure_label()
    fig_lc = rpt.next_figure_label(2)
    lc = RES["loss_curve"]
    lcv = RES["learning_curve"]
    rpt.para(
        f"{fig_loss} plots training loss and early-stopping validation accuracy on the first fold. Loss falls "
        f"smoothly from {lc['loss'][0]:.2f} to {lc['loss'][-1]:.2f}. Validation accuracy rises quickly, peaks "
        f"at {lc['best_val_accuracy']:.3f} in epoch {int(np.argmax(lc['val_accuracy'])) + 1}, then levels off. "
        f"Training stopped at epoch {lc['n_iter']} after 20 epochs with no improvement, and the best weights "
        "were restored. The smooth curve suggests the learning rate and batch size are reasonable; the plateau "
        "says extra epochs would mostly fit noise."
    )
    rpt.figure(FIG / "fig_loss_curve.png", "Training loss and early-stopping validation accuracy of the ANN on "
               "fold 1.", width=5.4, short="Training loss and validation accuracy")
    rpt.para(
        f"The learning curve in {fig_lc} shows validation ROC-AUC rising from {lcv['test_mean'][0]:.3f} with "
        f"{lcv['train_sizes'][0]} training rows to {lcv['test_mean'][-1]:.3f} with {lcv['train_sizes'][-1]} rows, "
        f"while training ROC-AUC stays near {np.mean(lcv['train_mean']):.2f}. The gap is small (about 0.02–0.05), "
        "so the regularised network is not over-fitting badly. The validation curve is still rising, which "
        "points to more data — not a bigger network — as the main way to do better."
    )
    rpt.figure(FIG / "fig_learning_curve.png", "Learning curve of the proposed ANN (mean ± sd over five folds).",
               width=5.0, short="Learning curve of the proposed ANN")
    rpt.section("4.6 Cross-Validated Comparison of All Configurations")
    tab_main = rpt.next_table_label()
    rpt.para(
        f"{tab_main} has protocol P1 results for all 13 configurations: the dummy floor, plus logistic "
        "regression, random forest, and the ANN, each with all eight features or SelectKBest (k = 5), and each "
        f"with and without training-fold oversampling. {rpt.next_figure_label()} is the console output of this "
        "run, with the same numbers and their fold standard deviations."
    )
    names = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "mlp_ann": "MLP ANN",
    }
    rows = [["Model", "Inputs", "ROS", "Acc.", "Prec.", "Sens.", "Spec.", "F1", "ROC-AUC", "PR-AUC"]]
    bold_rows = []
    for k, (name, v) in enumerate(RES["main_cv"].items(), 1):
        s = v["summary"]
        if name == "dummy_most_frequent":
            row = ["Dummy (majority)", "—", "—"]
        else:
            mdl, feats, rs = name.split("__")
            row = [names[mdl], "All 8" if feats == "all_features" else "k = 5", "Yes" if rs == "oversample" else "No"]
        row += [f"{s[m]['mean']:.3f}" for m in METRIC_ORDER]
        rows.append(row)
        if name == "mlp_ann__all_features__oversample":
            bold_rows.append(k)
    rpt.table(rows, "Protocol P1: seeded 5-fold cross-validation of all configurations (mean over folds; bold = "
              "proposed system)", widths=[1.2, 0.62, 0.48, 0.5, 0.5, 0.5, 0.5, 0.45, 0.62, 0.6], size=10,
              align=["l", "c", "c"] + ["c"] * 7, bold_rows=tuple(bold_rows),
              short="Seeded 5-fold comparison of all configurations")
    rpt.figure(FIG / "console_cv_main.png", "Console output of the experiment script: data loading and the "
               "seeded 5-fold comparison of 13 configurations (mean ± sd over folds).", width=6.0,
               short="Console output: seeded 5-fold comparison")
    mc = RES["main_cv"]
    rpt.para(
        "A few things jumped out. Every trained model beats the dummy on ROC-AUC (0.79–0.84 versus 0.50). "
        "The three families then sit in a tight band (0.826–0.842). Ranking, on this table, is mostly a "
        "property of the data. The ANN trained on the raw imbalance is shy: highest specificity "
        f"({mc['mlp_ann__all_features__no_resample']['summary']['specificity']['mean']:.3f}) but it only finds "
        f"{mc['mlp_ann__all_features__no_resample']['summary']['recall_sensitivity']['mean']:.1%} of the "
        "diabetic rows. Oversample the training fold and sensitivity goes to "
        f"{mc['mlp_ann__all_features__oversample']['summary']['recall_sensitivity']['mean']:.3f}. F1 goes from "
        f"{mc['mlp_ann__all_features__no_resample']['summary']['f1']['mean']:.3f} to "
        f"{mc['mlp_ann__all_features__oversample']['summary']['f1']['mean']:.3f}. That is why we kept "
        "oversampling. Logistic regression and the forest already use class weights, so oversampling barely "
        "touches them."
    )
    rpt.section("4.7 Repeated Cross-Validation and Statistical Significance")
    tab_rep = rpt.next_table_label()
    rpt.para(
        f"A single 5-fold run moves around with the split, so the five main models were re-run under protocol "
        f"P2. {tab_rep} gives mean ± standard deviation over 50 splits, and {rpt.next_table_label(2)} the "
        "corrected resampled t-tests."
    )
    rep_names = list(RES["repeated_cv"])
    rows = [["Metric", "Dummy", "Logistic reg.", "Random forest", "ANN (no ROS)", "Proposed ANN"]]
    for m in METRIC_ORDER:
        rows.append([METRIC_NAME[m]] + [ms(RES["repeated_cv"][n]["summary"], m) for n in rep_names])
    rpt.table(rows, "Protocol P2: 10 × 5 repeated stratified cross-validation (mean ± sd over 50 splits)",
              widths=[0.9, 0.9, 1.05, 1.05, 1.05, 1.05], size=10, align=["l"] + ["c"] * 5,
              short="Repeated cross-validation of the main models")
    rows = [["Comparison (proposed − other)", "Δ ROC-AUC (p)", "Δ PR-AUC (p)", "Δ F1 (p)", "Δ Sensitivity (p)"]]
    for other in ["Logistic regression", "Random forest", "MLP ANN (no rebalancing)"]:
        rows.append([f"vs {other}"] + [tcell(ttest(other, m)) for m in
                                        ["roc_auc", "pr_auc", "f1", "recall_sensitivity"]])
    rpt.table(rows, "Corrected resampled t-tests (Nadeau–Bengio) of the proposed ANN against each comparator",
              widths=[2.0, 1.0, 1.0, 1.0, 1.0], size=10, align=["l", "c", "c", "c", "c"],
              short="Corrected resampled t-tests")
    t_lr = ttest("Logistic regression", "roc_auc")
    t_rf = ttest("Random forest", "roc_auc")
    t_ns = ttest("MLP ANN (no rebalancing)", "recall_sensitivity")
    t_nf = ttest("MLP ANN (no rebalancing)", "f1")
    t_lr_s = ttest("Logistic regression", "recall_sensitivity")
    rpt.para(
        f"The proposed ANN reaches ROC-AUC {ms(RES['repeated_cv'][PROPOSED]['summary'], 'roc_auc')}, basically "
        f"the same as logistic regression ({ms(RES['repeated_cv']['Logistic regression']['summary'], 'roc_auc')}; "
        f"Δ = {t_lr['mean_diff']:+.3f}, {pval(t_lr['p'])}) and the random forest "
        f"({ms(RES['repeated_cv']['Random forest']['summary'], 'roc_auc')}; {pval(t_rf['p'])}). None of those "
        "ranking gaps is statistically significant. At the default threshold the ANN has the highest "
        f"sensitivity of the trained models ({rep(PROPOSED, 'recall_sensitivity'):.3f}), {t_lr_s['mean_diff']:.3f} "
        f"above logistic regression ({pval(t_lr_s['p'])}), with a matching drop in specificity. The one clear "
        f"effect is oversampling: against the same network on imbalanced folds, sensitivity rises by "
        f"{t_ns['mean_diff']:.3f} ({pval(t_ns['p'])}) and F1 by {t_nf['mean_diff']:.3f} "
        f"({pval(t_nf['p'])}), while ROC-AUC stays put. Oversampling therefore moves the decision boundary; it "
        f"does not make the ranking better. {rpt.next_figure_label()} plots these scores, and "
        f"{rpt.next_figure_label(2)} is the console output the tables were taken from."
    )
    rpt.figure(FIG / "fig_repeated_cv_bars.png", "Repeated cross-validation scores of the five main models "
               "(mean ± sd over 50 splits).", short="Repeated cross-validation scores")
    rpt.figure(FIG / "console_repeated_cv.png", "Console output of the repeated cross-validation and the "
               "corrected t-tests.", width=6.0, short="Console output: repeated CV and t-tests")
    rpt.section("4.8 ROC, Precision–Recall, and Confusion Matrices")
    fig_roc = rpt.next_figure_label()
    fig_cm = rpt.next_figure_label(2)
    cmp = RES["main_cv"]["mlp_ann__all_features__oversample"]["confusion_matrix_oof"]
    cmn = RES["main_cv"]["mlp_ann__all_features__no_resample"]["confusion_matrix_oof"]
    rpt.para(
        f"{fig_roc} shows out-of-fold ROC and precision–recall curves under protocol P1. The four trained "
        "models' curves sit on top of each other for most of the range, so their ranking ability is similar. "
        "The precision–recall panel is harsher: precision drops below 0.6 once recall goes past about 0.8 for "
        "every model. Finding most of the diabetic cases without a pile of false alarms is hard on this table. "
        "The dotted line is the PR baseline of 0.349, the positive rate."
    )
    rpt.figure(FIG / "fig_roc_pr_oof.png", "Out-of-fold ROC curves (left) and precision–recall curves (right) "
               "under seeded 5-fold cross-validation.", short="Out-of-fold ROC and precision–recall curves")
    rpt.para(
        f"The out-of-fold confusion matrices in {fig_cm} make those operating points concrete. The ANN trained "
        f"without rebalancing misses {cmn[1][0]} of 268 diabetic cases. The proposed ANN with training-fold "
        f"oversampling misses {cmp[1][0]} and finds {cmp[1][1]}, at the cost of {cmp[0][1]} false alarms among "
        f"500 negatives instead of {cmn[0][1]}. If a missed case is more costly than an extra confirmatory "
        "test — which is the usual screening trade-off — the proposed operating point is the more useful one."
    )
    rpt.figure(FIG / "fig_confusion_oof.png", "Out-of-fold confusion matrices (counts and row percentages) of "
               "the four trained models.", short="Out-of-fold confusion matrices")
    rpt.section("4.9 Decision-Threshold Analysis")
    yb = RES["threshold_sweep"]["youden_best"]
    rpt.para(
        f"{rpt.next_figure_label()} sweeps the decision threshold τ over the proposed ANN's out-of-fold "
        "probabilities. Sensitivity falls and specificity rises as τ goes up, while F1 stays between 0.66 and "
        f"0.69 for τ from 0.3 to 0.6. Youden's index peaks at τ = {yb['threshold']:.2f} (sensitivity "
        f"{yb['recall_sensitivity']:.3f}, specificity {yb['specificity']:.3f}). That threshold was read off the "
        "evaluation predictions, so it is a bit optimistic. It is shown only to make the trade-off visible; "
        "the headline results keep the pre-declared τ = 0.5. In practice the threshold should be set from the "
        "relative costs of missed cases and false alarms, on data that were not used for evaluation."
    )
    rpt.figure(FIG / "fig_threshold_sweep.png", "Sensitivity, specificity, precision, and F1 of the proposed "
               "ANN as a function of the decision threshold (out-of-fold).", width=5.4,
               short="Decision-threshold analysis")
    rpt.section("4.10 Hold-Out Test Evaluation")
    tab_ho = rpt.next_table_label()
    rpt.para(
        f"Protocol P4 trains each model once on 614 rows and scores it once on 154 untouched rows (54 of them "
        f"positive). {tab_ho} has the numbers, and {rpt.next_figure_label()} has the ROC curves plus the "
        "proposed ANN's confusion matrix."
    )
    rows = [["Model", "Acc.", "Prec.", "Sens.", "Spec.", "F1", "ROC-AUC", "PR-AUC"]]
    for name, v in ho.items():
        rows.append([name] + [f"{v['metrics'][m]:.3f}" for m in METRIC_ORDER])
    ci = ho_p["ci95"]
    rows.append(["Proposed ANN, 95% CI", f"{ci['accuracy'][0]:.2f}–{ci['accuracy'][1]:.2f}", "—",
                 f"{ci['recall_sensitivity'][0]:.2f}–{ci['recall_sensitivity'][1]:.2f}",
                 f"{ci['specificity'][0]:.2f}–{ci['specificity'][1]:.2f}",
                 f"{ci['f1'][0]:.2f}–{ci['f1'][1]:.2f}", f"{ci['roc_auc'][0]:.2f}–{ci['roc_auc'][1]:.2f}", "—"])
    rpt.table(rows, "Protocol P4: stratified 80/20 hold-out test (154 rows) with bootstrap 95% intervals for the "
              "proposed ANN", widths=[1.85, 0.55, 0.55, 0.55, 0.55, 0.5, 0.7, 0.65], size=10,
              align=["l"] + ["c"] * 7, bold_rows=(4,), short="Hold-out test results")
    rpt.figure(FIG / "fig_holdout.png", "Hold-out ROC curves of the four trained models (left) and the "
               "confusion matrix of the proposed ANN (right).", short="Hold-out ROC curves and confusion matrix")
    rpt.para(
        f"On the hold-out set the proposed ANN has the highest accuracy ({ho_p['metrics']['accuracy']:.3f}), "
        f"sensitivity ({ho_p['metrics']['recall_sensitivity']:.3f}), and F1 ({ho_p['metrics']['f1']:.3f}) of the "
        f"four models, and it ties the random forest on ROC-AUC ({ho_p['metrics']['roc_auc']:.3f}). It correctly "
        f"flags {tp} of {tp + fn} diabetic patients and {tn} of {tn + fp} non-diabetic patients. The bootstrap "
        "intervals are wide (sensitivity "
        f"{ci['recall_sensitivity'][0]:.2f}–{ci['recall_sensitivity'][1]:.2f}, for example) because there are "
        "only 54 positives, so the small gaps between models on this one split are not conclusive. The "
        "repeated-CV results in Section 4.7 are the safer comparison. The hold-out numbers still sit close to "
        "the cross-validated estimates, which is a check that the CV protocol is not painting a rosy picture."
    )
    fig_ho_console = rpt.figure(FIG / "console_holdout.png", "Console output of the hold-out evaluation: "
                                "per-model metrics, the classification report of the proposed ANN, and "
                                "permutation importances.", width=5.6, short="Console output: hold-out evaluation")
    rpt.para(
        f"The classification report in {fig_ho_console} breaks the same result down by class. For the diabetic "
        f"class, precision is {tp / (tp + fp):.3f} and recall {tp / (tp + fn):.3f}; for the non-diabetic class, "
        f"precision is {tn / (tn + fn):.3f} and recall {tn / (tn + fp):.3f}. The model is better at ruling "
        "diabetes out (a negative call is right about 87% of the time) than at confirming it. That is what a "
        "screening-style operating point is supposed to do."
    )
    rpt.section("4.11 Interpretability: Permutation Importance")
    pi = RES["permutation_importance"]
    rpt.para(
        f"{rpt.next_figure_label()} shows how much the hold-out ROC-AUC drops when each feature is shuffled. "
        f"Glucose is clearly the most important input (drop {pi['Glucose']['mean']:.3f}), then age "
        f"({pi['Age']['mean']:.3f}), BMI ({pi['BMI']['mean']:.3f}), and pregnancies ({pi['Pregnancies']['mean']:.3f}). "
        f"Blood pressure contributes least ({pi['BloodPressure']['mean']:.3f}). That order matches the ANOVA "
        "scores, the selection frequencies, the correlations, and the predictors picked by Joshi and Dhakal "
        f"{c('joshi2021')} and Howlader et al. {c('howlader2022')}. It is a useful check that the network is "
        "using signals a clinician would recognise, not artefacts of the table."
    )
    rpt.figure(FIG / "fig_permutation_importance.png", "Permutation importance of the proposed ANN on the "
               "hold-out set (mean ± sd over 30 shuffles).", width=5.0, short="Permutation importance")
    rpt.section("4.12 System Output Demonstration")
    rpt.para(
        f"{rpt.next_figure_label()} shows the prediction module in use. The saved pipeline (zero recoding → "
        "median imputation → scaling → oversampling → ANN; oversampling is idle at prediction time) is loaded "
        "and applied to the first ten hold-out patients. It prints the inputs, the predicted probability "
        "P(diabetes), the predicted class, the true class, and whether the call was right. Of those ten rows, "
        "six are correct, three are false alarms, and one diabetic case is missed. Across all 154 hold-out "
        "rows the module is right for 115 (74.7%). Two made-up records follow: a 47-year-old with glucose "
        "168 mg/dL and BMI 36.4 gets P = 0.914; a 26-year-old with glucose 94 mg/dL and BMI 24.1 gets P = 0.018. "
        "Zeros entered for skinfold and insulin in the first record are treated as missing and filled in, the "
        "same way as in training."
    )
    rpt.figure(FIG / "console_predict_demo.png", "Output of the prediction module on hold-out patients and two "
               "new records.", width=6.0, short="Output of the prediction module")
    rpt.para(
        f"{rpt.next_figure_label()} is the automated test suite. The 20 tests cover loading (ordinary, "
        "header-less, and alias-named CSV files, plus the ARFF converter), profiling of zeros and duplicates, "
        "the zero policy (Pregnancies never recoded), leakage safety (imputation statistics come from the "
        "training fold only, feature selection sits inside the pipeline, and resampling uses an "
        "imbalanced-learn pipeline so it runs only during fitting), metrics on edge cases, and an end-to-end "
        "run on a small table. All 20 pass."
    )
    rpt.figure(FIG / "console_pytest.png", "Unit-test output (20 tests passed).", width=5.6,
               short="Unit-test output")
    rpt.section("4.13 Discussion")
    rpt.para(
        "That answers the questions from Chapter 1, more or less. A small regularised ANN, trained without "
        f"leaking, sits around ROC-AUC {rep(PROPOSED, 'roc_auc'):.2f}. With training-fold oversampling the "
        f"operating point is about sensitivity {rep(PROPOSED, 'recall_sensitivity'):.2f} and specificity "
        f"{rep(PROPOSED, 'specificity'):.2f}. Same neighbourhood as the old 76%/76% ADAP cut. Indistinguishable "
        "from logistic regression and the forest on ranking. Extra ANN flexibility did not buy a better rank "
        "here. Glucose, BMI and age already carry most of the signal, and they are mostly monotonic. A linear "
        "model sees them. The useful bit is the screening-style cut, and that came from oversampling, not from "
        "the extra layers. The learning curve says the same thing. More rows would help. A fatter network "
        "probably would not."
    )

    # ============================================================ CHAPTER 5
    rpt.chapter_heading("Ablation Study and State-of-the-Art Analysis")
    rpt.section("5.1 Ablation Protocol")
    rpt.para(
        "Each ablation takes one piece out of the proposed system, or swaps it, and leaves everything else "
        "alone. All of them are scored with protocol P3 (3 × 5 repeated stratified CV, the same splits for "
        "every configuration). Gaps smaller than about one standard deviation (≈ 0.03–0.04 in ROC-AUC and F1) "
        "should be read as no clear effect. The ablation is descriptive: the proposed setup was fixed before "
        "these runs and was not re-tuned afterwards."
    )
    rpt.section("5.2 Component Ablation")
    rpt.para(
        f"{rpt.next_table_label()} removes or replaces one pipeline component at a time, "
        f"{rpt.next_figure_label()} plots the resulting F1 and ROC-AUC, and {rpt.next_figure_label(2)} shows the "
        "corresponding console output with standard deviations."
    )
    comp = [v for v in RES["ablation"].values() if v["group"] == "component"]
    base = comp[0]["summary"]
    rows = [["Configuration", "Sens.", "Spec.", "F1", "ROC-AUC", "PR-AUC", "Δ F1", "Δ ROC-AUC"]]
    for k, v in enumerate(comp):
        s = v["summary"]
        rows.append([v["label"]] + [f"{s[m]['mean']:.3f}" for m in
                                    ["recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]] +
                    (["—", "—"] if k == 0 else [f"{s['f1']['mean'] - base['f1']['mean']:+.3f}",
                                                f"{s['roc_auc']['mean'] - base['roc_auc']['mean']:+.3f}"]))
    rpt.table(rows, "Component ablation of the proposed pipeline (protocol P3, mean over 15 splits)",
              widths=[2.0, 0.5, 0.5, 0.5, 0.65, 0.6, 0.55, 0.7], size=10, align=["l"] + ["c"] * 7,
              bold_rows=(1,), short="Component ablation")
    rpt.figure(FIG / "fig_ablation_components.png", "Component ablation: F1 (left) and ROC-AUC (right); red bars "
               "fall more than 0.02 below the proposed system.", short="Component ablation chart")
    a = lambda lab, m: abl("component", lab, m)  # noqa: E731
    rpt.para(
        f"**Standardisation** is the piece that matters most. Without it ROC-AUC falls from "
        f"{base['roc_auc']['mean']:.3f} to {a('− standard scaling', 'roc_auc'):.3f} and F1 from "
        f"{base['f1']['mean']:.3f} to {a('− standard scaling', 'f1'):.3f}. Raw insulin values in the hundreds "
        "swamp the first-layer activations and the gradient steps, so the network barely sees small-scale "
        f"inputs such as the pedigree function. **Early stopping** is next: training for the full 500 epochs "
        f"lowers ROC-AUC to {a('− early stopping', 'roc_auc'):.3f}. That is over-fitting, as expected for a "
        "289-parameter network on about 700 rows. **Zero recoding** also matters. Keeping the zeros as real "
        f"values lowers ROC-AUC by {base['roc_auc']['mean'] - a('− zero recoding (zeros kept)', 'roc_auc'):.3f} "
        f"and F1 by {base['f1']['mean'] - a('− zero recoding (zeros kept)', 'f1'):.3f}, because a coded zero "
        "makes, for example, an insulin reading look like an extreme low."
    )
    rpt.para(
        "**Oversampling** does not change ranking (ROC-AUC "
        f"{a('− oversampling (imbalanced)', 'roc_auc'):.3f} without it) but it does set the operating point. "
        f"Without it, sensitivity falls to {a('− oversampling (imbalanced)', 'recall_sensitivity'):.3f} and F1 "
        f"to {a('− oversampling (imbalanced)', 'f1'):.3f}. Swapping random oversampling for **SMOTE** gives "
        "almost the same numbers, so the simpler method is kept. **Mean imputation** does as well as the "
        f"median (ROC-AUC {a('Median → mean imputation', 'roc_auc'):.3f}). **KNN imputation** is a little "
        f"worse ({a('Median → KNN imputation', 'roc_auc'):.3f}), probably because its distances are computed "
        "on unscaled features. Recoding **Pregnancies = 0** as missing slightly lowers ROC-AUC "
        f"({a('Recode Pregnancies = 0 as well', 'roc_auc'):.3f}), which is why valid zeros are kept. Adding "
        f"**SelectKBest (k = 5)** drops ROC-AUC to {a('+ SelectKBest (k = 5)', 'roc_auc'):.3f}. With only eight "
        "inputs, throwing three away loses some signal and gives the network nothing back."
    )
    rpt.figure(FIG / "console_ablation.png", "Console output of the component ablation (mean ± sd over 15 "
               "splits).", width=6.0, short="Console output: component ablation")
    rpt.section("5.3 Architecture and Hyperparameter Ablation")
    rpt.para(
        f"{rpt.next_table_label()} and {rpt.next_figure_label()} change one property of the network at a time "
        "(hidden layers, activation, optimiser, L2 penalty, and number of selected features) while the rest of "
        "the proposed pipeline stays put."
    )
    rows = [["Group", "Setting", "Sens.", "Spec.", "F1", "ROC-AUC", "PR-AUC"]]
    group_names = {"architecture": "Hidden layers", "activation": "Activation", "optimiser": "Optimiser",
                   "alpha": "L2 penalty α", "k_features": "Features kept k"}
    proposed_settings = {("architecture", "16 → 8"), ("activation", "relu"), ("optimiser", "Adam"),
                         ("alpha", "0.001"), ("k_features", "8")}
    bold = []
    for g, gname in group_names.items():
        for v in RES["ablation"].values():
            if v["group"] != g:
                continue
            s = v["summary"]
            label = v["label"] + (" *" if (g, v["label"]) in proposed_settings else "")
            rows.append([gname, label] + [f"{s[m]['mean']:.3f}" for m in
                                          ["recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]])
            if (g, v["label"]) in proposed_settings and g != "k_features":
                bold.append(len(rows) - 1)
            gname = ""
    rpt.table(rows, "Architecture and hyperparameter ablation (protocol P3; * = setting used by the proposed "
              "system; k = 8 equals no selection)", widths=[1.15, 1.25, 0.6, 0.6, 0.6, 0.7, 0.7], size=10,
              align=["l", "l", "c", "c", "c", "c", "c"], bold_rows=tuple(bold), short="Architecture and hyperparameter ablation")
    rpt.figure(FIG / "fig_ablation_hyper.png", "Architecture and hyperparameter ablation: ROC-AUC and F1 for each "
               "setting (mean ± sd over 15 splits).", short="Architecture and hyperparameter ablation chart")
    h = lambda g, lab, m: abl(g, lab, m)  # noqa: E731
    rpt.para(
        f"**Depth and width.** A single hidden layer of 8 units is a bit weaker (ROC-AUC "
        f"{h('architecture', '8', 'roc_auc'):.3f}), and the deepest network (64–32–16) is weaker too "
        f"({h('architecture', '64 → 32 → 16', 'roc_auc'):.3f}) because it over-fits. Networks with 16 to 64 "
        "hidden units in one or two layers all sit within 0.01 of each other (0.829–0.837). The proposed 16–8 "
        "network is on that plateau; it was not cherry-picked. **Activation.** ReLU and tanh are equivalent "
        f"({h('activation', 'relu', 'roc_auc'):.3f} and {h('activation', 'tanh', 'roc_auc'):.3f}). The logistic "
        f"activation trains more slowly and does worse ({h('activation', 'logistic', 'roc_auc'):.3f}), which "
        f"matches the vanishing-gradient argument for saturating units {c('glorot2010')}. **Optimiser.** SGD "
        f"with momentum matches Adam ({h('optimiser', 'SGD + momentum', 'roc_auc'):.3f}), but L-BFGS, which in "
        "scikit-learn does not use early stopping, over-fits badly "
        f"({h('optimiser', 'L-BFGS', 'roc_auc'):.3f}). That is another reminder that early stopping is doing "
        "real work."
    )
    rpt.para(
        f"**L2 penalty.** ROC-AUC is flat (0.836–0.838) for α between 10^{{−5}} and 1. At α = 1 the network "
        "becomes more conservative and trades specificity for sensitivity "
        f"({h('alpha', '1', 'recall_sensitivity'):.3f}). With early stopping already in place, L2 is a "
        "secondary regulariser here. **Number of features.** Every k < 8 gives a ROC-AUC at or below the "
        f"all-feature result ({h('k_features', '8', 'roc_auc'):.3f}). The values for k = 3 to 7 "
        f"({h('k_features', '7', 'roc_auc'):.3f}–{h('k_features', '6', 'roc_auc'):.3f}) differ by less than one "
        "standard deviation, so no subset beats using all eight inputs. Across this grid the proposed design "
        "is robust: nothing in the range is better by more than the noise."
    )
    rpt.section("5.4 Leakage Ablation")
    tab_leak = rpt.next_table_label()
    rpt.para(
        "To see how protocol shortcuts inflate the numbers, the proposed ANN and the random forest were run "
        "under four deliberately leaky protocols and compared with the leakage-safe protocol on the same "
        f"splits ({tab_leak}): (i) fitting the imputer and scaler on all 768 rows before CV; (ii) also "
        "selecting five features with SelectKBest on all rows; (iii) oversampling all rows before CV; and "
        "(iv) applying SMOTE to all rows before CV."
    )
    rows = [["Model", "Protocol", "F1", "ROC-AUC", "PR-AUC", "Δ F1", "Δ PR-AUC"]]
    for model in ["MLP", "RF"]:
        safe = RES["leakage"][f"{model}::Leakage-safe (proposed)"]
        first = True
        for key, v in RES["leakage"].items():
            if not key.startswith(model + "::"):
                continue
            lab = key.split("::")[1]
            delta = lab != "Leakage-safe (proposed)"
            rows.append(["ANN (MLP)" if (model == "MLP" and first) else ("Random forest" if first else ""), lab,
                         f"{v['f1']['mean']:.3f}", f"{v['roc_auc']['mean']:.3f}", f"{v['pr_auc']['mean']:.3f}",
                         f"{v['f1']['mean'] - safe['f1']['mean']:+.3f}" if delta else "—",
                         f"{v['pr_auc']['mean'] - safe['pr_auc']['mean']:+.3f}" if delta else "—"])
            first = False
    rpt.table(rows, "Leakage ablation: leakage-safe protocol versus four leaky protocols (protocol P3)",
              widths=[1.05, 1.8, 0.55, 0.7, 0.65, 0.55, 0.7], size=10, align=["l", "l"] + ["c"] * 5,
              short="Leakage ablation")
    rpt.figure(FIG / "fig_leakage.png", "Effect of leaky protocols on F1, ROC-AUC, and PR-AUC for the ANN (left) "
               "and the random forest (right); hatched bars are leaky.", short="Leakage ablation chart")
    rpt.para(
        "Fitting the imputer and scaler on the full table barely moves the scores (|Δ| ≤ 0.01). Medians, "
        "means, and variances of 768 rows are almost the same as those of 614 rows. Global feature selection "
        "also does little here, because the strongest features are obvious. Resampling before splitting is "
        "another story. Duplicated (or interpolated) minority patients then sit in both the training and test "
        f"folds, and the models get credit for recognising them ({rpt.figures[-1][0]}). For the ANN, "
        f"oversampling before CV raises F1 by {leak('MLP::Oversample before CV', 'f1') - leak('MLP::Leakage-safe (proposed)', 'f1'):.3f} "
        f"and PR-AUC by {leak('MLP::Oversample before CV', 'pr_auc') - leak('MLP::Leakage-safe (proposed)', 'pr_auc'):.3f}. "
        f"For the random forest, which can memorise duplicates in its leaves, SMOTE before CV raises F1 by "
        f"{leak('RF::SMOTE before CV', 'f1') - leak('RF::Leakage-safe (proposed)', 'f1'):.3f}, ROC-AUC by "
        f"{leak('RF::SMOTE before CV', 'roc_auc') - leak('RF::Leakage-safe (proposed)', 'roc_auc'):.3f}, and PR-AUC "
        f"by {leak('RF::SMOTE before CV', 'pr_auc') - leak('RF::Leakage-safe (proposed)', 'pr_auc'):.3f}. Those "
        "gains are artefacts of the protocol; nothing about the models changed. Resampling also balances the "
        "evaluation set, so the leaky accuracy and precision no longer describe the 65:35 population at all."
    )
    rpt.section("5.5 Nested versus Non-Nested Hyperparameter Selection")
    rows = [["Outer fold", "Selected hidden layers", "Selected α", "Inner CV ROC-AUC", "Outer-fold ROC-AUC"]]
    for i, (ch, auc) in enumerate(zip(nested["chosen_per_outer_fold"], nested["nested_fold_auc"]), 1):
        rows.append([str(i), ch["hidden"], f"{ch['alpha']:g}", f"{ch['inner_auc']:.3f}", f"{auc:.3f}"])
    best_h, best_a = nested["best_non_nested_key"].split("|")
    rpt.para(
        f"A 16-point grid over hidden layers {{(8), (16), (16, 8), (32, 16)}} and α {{10^{{−4}}, …, 10^{{−1}}}} was "
        "searched in two ways. The non-nested way scores every grid point with the same 5-fold CV and quotes "
        f"the best one: {best_h.replace(',)', ')')} with α = {best_a}, ROC-AUC "
        f"{nested['best_non_nested_auc']:.3f}. The nested way picks the grid point with an inner 5-fold CV on "
        f"each outer training fold and scores it on the outer test fold ({rpt.next_table_label()}). The nested "
        f"estimate is {nested['nested_auc_mean']:.3f} ± {nested['nested_auc_std']:.3f}, "
        f"{nested['best_non_nested_auc'] - nested['nested_auc_mean']:.3f} lower, and the chosen setup changes "
        "from fold to fold. That is the selection bias described by Varma and Simon "
        f"{c('varma2006')} and Cawley and Talbot {c('cawley2010')}, measured on this table. It is also why the "
        "proposed configuration was fixed in advance rather than picked from the results. The console output "
        f"of the leakage and nested experiments is in {rpt.next_figure_label()}."
    )
    rpt.table(rows, "Nested cross-validation: hyperparameters selected in each outer fold and the resulting "
              "outer-fold ROC-AUC", widths=[0.8, 1.5, 0.9, 1.3, 1.4], size=10, align=["c"] * 5,
              short="Nested cross-validation results")
    rpt.figure(FIG / "console_leakage_nested.png", "Console output of the leakage experiments and the nested "
               "versus non-nested comparison.", width=6.0, short="Console output: leakage and nested CV")
    rpt.section("5.6 Seed Stability")
    rpt.para(
        f"{rpt.next_figure_label()} repeats the proposed system's 5-fold CV with ten other seeds (0–9) for the "
        "network initialisation and the oversampler, keeping the folds fixed. ROC-AUC ranges from "
        f"{seed['roc_auc']['min']:.3f} to {seed['roc_auc']['max']:.3f} (mean {seed['roc_auc']['mean']:.3f} ± "
        f"{seed['roc_auc']['std']:.3f}) and sensitivity from {seed['recall_sensitivity']['min']:.3f} to "
        f"{seed['recall_sensitivity']['max']:.3f}. The default seed 42 gives ROC-AUC "
        f"{mc['mlp_ann__all_features__oversample']['summary']['roc_auc']['mean']:.3f}, which is higher than eight "
        "of the ten other seeds, so the seed-42 figures in this report sit about 0.01 above the seed-averaged "
        "expectation. The seed-to-seed spread (≈ 0.04 in ROC-AUC) is as large as most of the ablation "
        "differences. Only the large effects in Section 5.2 should be treated as real."
    )
    rpt.figure(FIG / "fig_seed_stability.png", "Seed stability of the proposed ANN: 5-fold mean scores for ten "
               "random seeds (dotted lines = seed means).", width=5.4, short="Seed stability of the proposed ANN")
    rpt.section("5.7 State-of-the-Art Comparison")
    tab_sota = rpt.next_table_label()
    rpt.para(
        f"{tab_sota} puts the proposed system next to published results on the same dataset. Accuracy is "
        "shown because it is the only metric most studies report. Sensitivity, specificity, and AUC are "
        "filled in where the authors quoted them. The evaluation-protocol column is the one that actually "
        "lets the numbers be compared."
    )
    s_rep = RES["repeated_cv"][PROPOSED]["summary"]
    sota = [
        ["Study", "Method", "Protocol", "Acc. (%)", "Sens.", "Spec.", "AUC"],
        [f"Smith et al. {c('smith1988')}", "ADAP neural model", "576/192 split", "—", "0.76", "0.76", "ROC shown"],
        [f"Kayaer & Yıldırım {c('kayaer2003')}", "GRNN", "Train/test split", "80.21", "—", "—", "—"],
        [f"Polat & Güneş {c('polat2007')}", "PCA + ANFIS", "10-fold CV", "89.47", "—", "—", "—"],
        [f"Kahramanli & Allahverdi {c('kahramanli2008')}", "Hybrid ANN + FNN", "k-fold CV", "84.24", "—", "—", "—"],
        [f"Temurtas et al. {c('temurtas2009')}", "MLNN (LM)", "Split / 10-fold CV", "82.37", "—", "—", "—"],
        [f"Erkaymaz et al. {c('erkaymaz2017')}", "Small-world FFANN", "k-fold CV", "93.06", "0.85", "0.96", "—"],
        [f"Maniruzzaman et al. {c('maniruzzaman2017')}", "Gaussian process", "K-fold CV", "81.97", "0.918", "0.633", "—"],
        [f"Sisodia & Sisodia {c('sisodia2018')}", "Naïve Bayes", "Cross-validation", "76.30", "—", "—", "—"],
        [f"Hasan et al. {c('hasan2020')}", "Weighted ensemble", "k-fold CV", "—", "0.789", "0.934", "0.950"],
        [f"Joshi & Dhakal {c('joshi2021')}", "Logistic regression", "Split + CV", "78.26", "—", "—", "—"],
        [f"Kumari et al. {c('kumari2021')}", "Soft-voting ensemble", "Train/test", "79.04", "—", "—", "—"],
        [f"Howlader et al. {c('howlader2022')}", "Gen. boosted regression", "Cross-validation", "90.91", "—", "—", "—"],
        [f"Chang et al. {c('chang2023')}", "Random forest", "70/30 split", "79.57", "—", "—", "0.862"],
        ["This work (proposed)", "Leakage-safe ANN + ROS", "10×5 repeated CV",
         f"{s_rep['accuracy']['mean'] * 100:.2f}", f"{s_rep['recall_sensitivity']['mean']:.3f}",
         f"{s_rep['specificity']['mean']:.3f}", f"{s_rep['roc_auc']['mean']:.3f}"],
        ["This work (proposed)", "Leakage-safe ANN + ROS", "80/20 hold-out",
         f"{ho_p['metrics']['accuracy'] * 100:.2f}", f"{ho_p['metrics']['recall_sensitivity']:.3f}",
         f"{ho_p['metrics']['specificity']:.3f}", f"{ho_p['metrics']['roc_auc']:.3f}"],
        ["This work (leaky, for reference)", "Random forest, SMOTE before CV", "3×5 repeated CV",
         f"{leak('RF::SMOTE before CV', 'accuracy') * 100:.2f}", f"{leak('RF::SMOTE before CV', 'recall_sensitivity'):.3f}",
         f"{leak('RF::SMOTE before CV', 'specificity'):.3f}", f"{leak('RF::SMOTE before CV', 'roc_auc'):.3f}"],
    ]
    rpt.table(sota, "State-of-the-art comparison on the Pima Indians Diabetes dataset (published figures as "
              "reported by the authors)", widths=[1.5, 1.25, 0.95, 0.6, 0.5, 0.5, 0.65], size=10,
              align=["l", "l", "l", "c", "c", "c", "c"], bold_rows=(len(sota) - 3, len(sota) - 2),
              short="State-of-the-art comparison")
    rpt.para(
        "The table falls into three groups. **(a) Studies with simple models and simple protocols** (logistic "
        "regression, naïve Bayes, random forests, soft voting, GRNN) quote 76–82% accuracy, the same range as "
        f"this work's {s_rep['accuracy']['mean'] * 100:.1f}% (repeated CV) and "
        f"{ho_p['metrics']['accuracy'] * 100:.1f}% (hold-out). The proposed ANN therefore sits with the honest "
        "state of the art on this benchmark. Its sensitivity–specificity pair is close to the 0.76/0.76 of "
        "Smith et al. and more balanced than the 0.918/0.633 of the Gaussian-process model. **(b) Studies "
        "quoting 84–93% accuracy or AUC ≈ 0.95** use heavier pipelines (PCA, fuzzy hybrids, rewired "
        "topologies, outlier removal, ensembles). On a 768-row table those numbers depend a lot on exactly "
        "when preprocessing, selection, and tuning are fitted. **(c) This work's leaky reference row** shows "
        "that a standard random forest reaches ROC-AUC "
        f"{leak('RF::SMOTE before CV', 'roc_auc'):.3f} and sensitivity "
        f"{leak('RF::SMOTE before CV', 'recall_sensitivity'):.3f} just by applying SMOTE before "
        "cross-validation."
    )
    rpt.para(
        "We are not accusing any one paper of leakage. We are saying protocol choices alone can move the "
        "numbers by about as much as the gap between group (a) and group (b). If a Pima write-up does not "
        "say how preprocessing, resampling and tuning sat inside the validation, accuracy is not a fair "
        "comparison. Against the papers that used ordinary, transparent protocols, our leakage-safe ANN sits "
        "in the same band. It also reports the stuff most of them skip: full metrics, tests against "
        "baselines, an ablation, and a direct estimate of how much the usual shortcuts inflate the score."
    )
    rpt.section("5.8 Summary of Findings")
    rpt.bullets([
        f"The proposed ANN reaches ROC-AUC {rep(PROPOSED, 'roc_auc'):.3f}, sensitivity "
        f"{rep(PROPOSED, 'recall_sensitivity'):.3f}, and specificity {rep(PROPOSED, 'specificity'):.3f} under "
        "repeated CV, and similar values on the hold-out set.",
        "In ranking quality it is statistically indistinguishable from logistic regression and a random "
        "forest. What it does offer is a screening-oriented operating point, created by training-fold "
        "oversampling.",
        "Standardisation, early stopping, and zero recoding are the pieces that matter. Architecture, "
        "activation (except logistic), L2 strength, and feature selection do not.",
        "Resampling before splitting inflates PR-AUC by 0.12–0.15 and F1 by 0.09–0.12. Non-nested tuning "
        f"inflates ROC-AUC by about {nested['best_non_nested_auc'] - nested['nested_auc_mean']:.2f}.",
        "Seed-to-seed variation (≈ 0.04 in ROC-AUC) is as large as most configuration differences, so only "
        "the large effects should be trusted.",
    ])

    # ============================================================ CHAPTER 6
    rpt.chapter_heading("Conclusion and Future Work")
    rpt.section("6.1 Conclusion")
    conclusion = [
        "We set out to put an ANN at the centre of a Diabetes Prediction System and to score it without "
        "cheating on the Pima table. The table is small. Classes are skewed. Zeros sit in columns where a "
        "living adult cannot have a zero. Published accuracies still run from the mid-70s to well over 90%, "
        "and that spread is what made us suspicious in the first place. So training a net was only half the "
        "job. The other half was asking what the net actually does when every learned step stays off the "
        "evaluation rows. If the honest score landed near logistic regression, that was still a result worth "
        "writing down.",
        "Impossible zeros in five columns are recoded. Valid zero pregnancies are left alone. Training-fold "
        "medians fill holes. Inputs are scaled. The minority class is oversampled only inside the training "
        "fold. Then an 8–16–8–1 MLP is trained with ReLU, a sigmoid, Adam, L2 and early stopping. We wrote "
        "the method out (equations and algorithms), put it in a tested Python package, and scored it with "
        "seeded CV, repeated CV, one hold-out, corrected tests and bootstrap intervals. None of that is "
        "fancy. It is just the stuff that has to sit inside the fold if you want the number to mean anything.",
        f"On 10 × 5 repeated CV we got ROC-AUC {rep(PROPOSED, 'roc_auc'):.3f}, sensitivity "
        f"{rep(PROPOSED, 'recall_sensitivity'):.3f}, specificity {rep(PROPOSED, 'specificity'):.3f}. Hold-out "
        f"found {tp} of {tp + fn} diabetic patients. Ranking was a dead heat with logistic regression and the "
        "random forest. That is not a failure. Eight mostly monotonic predictors do not leave a lot of extra "
        "signal for a neural net to grab. The practical difference was the operating point. Training-fold "
        "oversampling raised sensitivity by about 0.21 and left ranking alone, which is the cut you want if "
        "missed cases are costly. Permutation importance put glucose first, then age, BMI, pregnancies. That "
        "matches the usual type 2 story. No surprise there.",
        "Ablation sorted the priorities. Scaling, early stopping and the medical zero rule moved the scores. "
        "Depth, width, ReLU versus tanh, L2 strength and univariate selection sat inside seed noise. "
        "Oversample before CV and PR-AUC inflates 0.12 to 0.15; F1 by up to 0.12. Pick hyperparameters on "
        "the evaluation folds and you gift yourself about 0.02 ROC-AUC. Ten seeds already swing ROC-AUC by "
        "about 0.04. So we stopped trusting small gaps. Those measured bumps also explain why the published "
        "Pima numbers sit so far apart.",
        "Limits are not subtle. One community of adult women, exams from decades ago, a historical glucose "
        "rule, nearly half of insulin imputed, no external cohort. Educational risk estimate. Not a "
        "diagnostic device. Inside those limits a small, regularised ANN is a decent Pima classifier. The "
        "protocol deserves as much attention as which network you pick. We would not ship this to a clinic. "
        "We would use it to teach the pipeline: recode the zeros properly, keep the test fold clean, and "
        "stop quoting accuracy as if it told the whole story. That is the actual takeaway from the semester."
    ]
    for p in conclusion:
        rpt.para(p)
    conclusion_words = sum(word_count(p) for p in conclusion)
    rpt.section("6.2 Future Work")
    rpt.bullets([
        "Run the same pipeline on an external, mixed-sex, contemporary cohort collected with consent and "
        "ethics approval. Report calibration, not only ranking.",
        "Swap the single median fill for multiple imputation so the 48.7% missing insulin values show up as "
        "uncertainty in the predictions, not as a fake certainty.",
        "Set the decision threshold from actual costs of missed cases versus false alarms, on a validation "
        "set or with nested CV. Not from the evaluation predictions.",
        "Put a local explanation next to each call (SHAP would do) and check whether the net actually learns "
        "interactions a linear model cannot see.",
        "Wrap the trained pipeline as a small web or mobile demo with input checks and a clear "
        "non-diagnostic disclaimer. Lab use only.",
    ])

    # ------------------------------------------------------------ references
    rpt.unnumbered_heading("REFERENCES", toc=True, new_page=True)
    for k, key in enumerate(refs.order, 1):
        par = rpt.doc.add_paragraph()
        par.alignment = LEFT
        par.paragraph_format.left_indent = Inches(0.45)
        par.paragraph_format.first_line_indent = Inches(-0.45)
        par.paragraph_format.space_after = Pt(4)
        par.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
        add_runs(par, f"[{k}]\t{CATALOGUE[key]}")

    def clean(t: str) -> str:
        return re.sub(r"\*", "", t)

    rpt.fill_listing(toc_anchor, [(lvl, clean(t), f"H::{t}") for lvl, t in rpt.headings])
    rpt.fill_listing(lof_anchor, [(2, f"{lab.replace('Figure', 'Fig')}: {clean(cap)}", f"F::{lab}")
                                  for lab, cap in rpt.figures])
    rpt.fill_listing(lot_anchor, [(2, f"{lab}: {clean(cap)}", f"T::{lab}") for lab, cap in rpt.tables])
    rpt.stats = {"abstract_words": abstract_words, "conclusion_words": conclusion_words,
                 "references": len(refs.order), "figures": len(rpt.figures), "tables": len(rpt.tables),
                 "equations": None}
    return rpt


# ---------------------------------------------------------------------------
# PDF export and page mapping
# ---------------------------------------------------------------------------


def export_pdf(docx: Path, pdf: Path) -> None:
    if pdf.exists():
        pdf.unlink()
    subprocess.run(["osascript", str(ROOT / "export_pdf.applescript"), str(docx), str(pdf)], check=True)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\u00ad", "")).strip()


def page_map_from_pdf(pdf: Path, rpt: Report) -> dict[str, int]:
    doc = fitz.open(str(pdf))
    pages = [norm(p.get_text()) for p in doc]
    start = next(i for i, t in enumerate(pages) if t.startswith("1. INTRODUCTION"))
    out: dict[str, int] = {}

    def find(needle: str) -> int | None:
        n = norm(needle)
        for i in range(start, len(pages)):
            if n in pages[i]:
                return i - start + 1
        return None

    for _, text in rpt.headings:
        out[f"H::{text}"] = find(text)
    for lab, _ in rpt.figures:
        out[f"F::{lab}"] = find(f"{lab}:")
    for lab, _ in rpt.tables:
        out[f"T::{lab}"] = find(f"{lab}:")
    missing = [k for k, v in out.items() if v is None]
    if missing:
        print("WARNING: not found in PDF:", missing)
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    tmp_docx = BUILD / "pass1.docx"
    tmp_pdf = BUILD / "pass1.pdf"
    rpt = build({})
    rpt.save(tmp_docx)
    export_pdf(tmp_docx, tmp_pdf)
    pmap = page_map_from_pdf(tmp_pdf, rpt)

    for attempt in range(3):
        rpt = build(pmap)
        rpt.save(DOCX_OUT)
        export_pdf(DOCX_OUT, PDF_OUT)
        new_map = page_map_from_pdf(PDF_OUT, rpt)
        if new_map == pmap:
            break
        print(f"page map changed on pass {attempt + 2}; rebuilding")
        pmap = new_map

    pdf = fitz.open(str(PDF_OUT))
    front = next(i for i in range(pdf.page_count) if norm(pdf[i].get_text()).startswith("1. INTRODUCTION"))
    print(f"Wrote {DOCX_OUT.name} and {PDF_OUT.name}: {pdf.page_count} pages "
          f"({front} front-matter pages + {pdf.page_count - front} numbered pages)")
    print(json.dumps(rpt.stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
