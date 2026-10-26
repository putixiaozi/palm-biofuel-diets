"""Repository paths. FABIO version 2 is not redistributed: set FABIO_DIR to the folder obtained from its developers."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
FABIO = os.environ.get("FABIO_DIR", REPO + "data_fabio") .rstrip("/") + "/"
