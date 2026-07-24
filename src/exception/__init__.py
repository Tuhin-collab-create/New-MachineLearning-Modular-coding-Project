import os, sys

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        self.error_message = CustomException.get_detailed_error_message(
            error_message=error_message, error_details=error_details
        )
        super().__init__(self.error_message)
    
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys):
        _, _, exc_tb = error_details.exc_info()
        
        if exc_tb is not None:
            error_line = exc_tb.tb_lineno
            error_file = exc_tb.tb_frame.f_code.co_filename
            message = (
                f"Error occurred in script: [{error_file}] at line number: [{error_line}] error message: [{error_message}]"
            )
        else:
            message = str(error_message)
            
        return message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return self.__class__.__name__
    