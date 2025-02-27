import sys 
from src.logger import logging

def error_message_details(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_msg = "Error occured in python script: [{0}] in line number: [{1}] \n\
        Error message: [{2}]".format(file_name, line_no, str(error))
    return error_msg
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    


    