# Importing packages
import sys

# Creating a function to fetch the error message from the sys module
def fetch_error_message(error, error_detail:sys):
    '''
    This function fetches the error information from the sys module.
    ====================================================================
    ------------------
    Parameters:
    ------------------
    error - str : This is the error message.
    error_detail: This is the error detail from the sys module.
    
    ------------------
    Returns:
    ------------------
    error_message - This is the error message from the sys module.
    =====================================================================
    '''
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = "Error occurred in script name [{0}], line number [{1}], with error message [{2}]".format(file_name, line_no, str(error))
    return error_message


# Creating a custom exception class to display exceptions thrown by the system.
# This custom class will inherit from the Python Exception class.
class CustomException(Exception):
    '''
    The CustomException class is a custom class to display system exceptions. 
    This class inherits from the Python Exception class.
    The class contains two functions - The constructor and a function to display the error message.
    '''
    # Defining the class constructor
    def __init__(self, error_message, error_detail:sys):
        # Instantiating the parent class and passing the error message to it
        super().__init__(error_message)
        self.error_message = fetch_error_message(error_message, error_detail=error_detail)
    
    # Creating a function to print the error message on the screen
    def __str__(self):
        return self.error_message
