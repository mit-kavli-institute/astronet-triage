HUMAN_READABLE_LABELS = {"Planet", "Eclipsing Binary", "Noise", "Junk"}
HUMAN_LABEL_MAP = {
    "Planet": "disp_p",
    "Eclipsing Binary": "disp_e",
    "Unknown": "disp_n",
    "Junk": "disp_j",
}
PREDICTION_LABELS = ["disp_p", "disp_e", "disp_n", "disp_j"]
PREDICTION_MAPPING = {"disp_p": "Planet", "disp_e": "Eclipsing Binary", "disp_n": "Noise", "disp_j": "Junk"}

TRUE_MAPPING = {
    "eb": "Eclipsing Binary", "ebs": "Eclipsing Binary", "et": "Eclipsing Binary", "eu": "Eclipsing Binary",
    "ets": "Eclipsing Binary", "eus": "Eclipsing Binary", "pt": "Planet", "pb": "Planet", "pu": "Planet",
    "pts": "Planet", "pus": "Planet", "nt": "Noise", "nb": "Noise", "nu": "Noise", "jj": "Junk",
    "ub": "Unknown Binary", "i": "Indeterminate", "p": "Planet", "e": "Eclipsing Binary", "n": "Noise", "j": "Junk"
}