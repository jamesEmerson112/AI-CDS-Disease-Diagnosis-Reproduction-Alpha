class Drgcodes:

    ROW_ID = 0
    SUBJECT_ID = 1
    HADM_ID = 2
    DRG_TYPE = 3
    DRG_CODE = 4
    DESCRIPTION = 5
    DRG_SEVERITY = 6
    DRG_MORTALITY = 7

    def __init__(self, row_id, subject_id, hadm_id, drg_type, drg_code, description, drg_severity, drg_mortality):
        self.row_id = row_id
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.drg_type = drg_type
        self.drg_code = drg_code
        self.description = description
        self.drg_severity = drg_severity
        self.drg_mortality = drg_mortality
