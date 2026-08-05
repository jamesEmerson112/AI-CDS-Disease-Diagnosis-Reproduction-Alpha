class Admission:

    CONST_ROW_ID = 0
    CONST_SUBJECT_ID = 1
    CONST_HADM_ID = 2
    CONST_ADMITTIME = 3
    CONST_DISCHTIME = 4
    CONST_DEATHTIME = 5
    CONST_ADMISSION_TYPE = 6
    CONST_ADMISSION_LOCATION = 7
    CONST_DISCHARGE_LOCATION = 8
    CONST_INSURANCE = 9
    CONST_LANGUAGE = 10
    CONST_RELIGION = 11
    CONST_MARITAL_STATUS = 12
    CONST_ETHNICITY = 13
    CONST_EDREGTIME = 14
    CONST_EDOUTTIME = 15
    CONST_DIAGNOSIS = 16
    CONST_HOSPITAL_EXPIRE_FLAG = 17
    CONST_HAS_CHARTEVENTS_DATA = 18

    def __init__(self, row_id, subject_id, hadm_id, admittime, dischtime,
               deathtime, admission_type, admission_location, discharge_location,
               insurance, language, religion, marital_status, ethnicity, edregtime,
               edouttime, diagnosis, hospital_expire_flag, has_chartevents_data):
      self.row_id = row_id
      self.subject_id = subject_id
      self.hadm_id = hadm_id
      self.admittime = admittime
      self.dischtime = dischtime
      self.deathtime = deathtime
      self.admission_type = admission_type
      self.admission_location = admission_location
      self.discharge_location = discharge_location
      self.insurance = insurance
      self.language = language
      self.religion = religion
      self.marital_status = marital_status
      self.ethnicity = ethnicity
      self.edregtime = edregtime
      self.edouttime = edouttime
      self.diagnosis = diagnosis
      self.hospital_expire_flag = hospital_expire_flag
      self.has_chartevents_data = has_chartevents_data



