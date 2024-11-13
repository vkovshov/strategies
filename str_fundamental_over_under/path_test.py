import sys
import os
sys.path.append(os.path.abspath('fin_data'))
print(sys.path)
from utils.postgresql_conn import get_session
