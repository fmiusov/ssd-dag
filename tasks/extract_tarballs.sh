for filename in /home/jay/projects/ssd-dag/data/new_jpeg_tarballs/*.tar.gz
do
  echo $filename
  tar -xvf $filename -C /home/jay/projects/ssd-dag/data/new_jpegs_tarball_extract/
done

