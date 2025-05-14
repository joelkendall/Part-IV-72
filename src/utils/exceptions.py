class TSVError(Exception):
    """Base class for all TSV-related errors."""
    pass


class TSVFileNotFoundError(TSVError):
    """Error raised when a file is not found."""
    pass

class TSVMultipleFilesFoundError(TSVError):
    """Error raised when multiple files are found."""
    pass

class TSVWriteError(TSVError):
    """Error raised when there is an error writing to a TSV file."""
    pass

