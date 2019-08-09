import tarfile

def extract_tarball(tarball, output_path):
    tb = tarfile.open(tarball)
    tb.extractall(path=output_path)
