import sys


def update_progress(progress, message='Percent'):
    """
    Lightweight function to (create and) update progress bar for processes used throughout this package.
    No depenencies.

    :param progress:    Progression as a fraction of steps completed
    :param message:     Message to accompany progress bar
    """

    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\r"+message+": [{0}] {1:.3f}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
