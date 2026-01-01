class SymptomsDiagnosis:

    CONST_HADM_ID = 0
    CONST_SUBJECT_ID = 1
    CONST_ADMITTIME = 2
    CONST_DISCHTIME = 3
    CONST_SYMPTOMS = 4
    CONST_DIAGNOSIS = 5

    def __init__(self, hadm_id, subject_id, admittime, dischtime, symptoms, diagnosis):
        self.hadm_id = hadm_id
        self.subject_id = subject_id
        self.admittime = admittime
        self.dischtime = dischtime
        self.symptoms = symptoms
        self.diagnosis = diagnosis