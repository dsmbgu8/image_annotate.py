# Simple interactive image annotation tool.

## Requirements:
  - matplotlib
  - numpy
  - scipy (for loading Matlab files)
  - wxPython (for WXagg matplotlib backend)

## Usage:
```
python image_annotate.py [options] [image_file]
  -h  --help         print this usage and exit
  -r  --ref          reference image (RGB or RGBA image to display)
  -b  --bands        indices of reference bands (default=[0,1,2])
  -w  --wvl          wavelength file
  -l  --load-rois    load ROIs from file
  -c  --color        color file (default=cm_jet16.txt)
  -n  --norm         L2 normalize each pixel in image_file
  -s  --scale        scaling factor for reference image colors (default=1.0)
  -f  --flip         flip data up/down w.r.t. reference file
  -v  --verbose      enable verbose output 
  ```
  
## Mouse commands:
```
  left click:    define new ROI point, drag for free-hand drawing, 
                 select/deselect existing ROI
                 
  right click:   close current ROI polygon, delete existing ROI
  ```
  
## Keyboard commands:
```
  e: export rois
  m: export mean image
  a: annotate ROIs
  d: delete last ROI
  g: group selected ROIs
  u: ungroup selected ROIs
  j: join unlabeled ROIs to labeled ROI group
  c: clear all unlabeled ROIs
  =: grow zoom window 
  -: shrink zoom window
  q: quit (and export if ROIs changed since startup)
  ```
  
## Loading matlab (.mat) files as images:
Because `scipy.io.loadmat` returns a dictonary, it is necessary to
provide the key to that dictionary where the image data is located. To
do so, use the following syntax

```bash
python image_annotate.py [options] /path/to/image.mat:key_value
```

This syntax also works for the ref and wvl files. For instance, the
command:

```bash
python image_annotate.py -r /path/to/ref.mat:ref_img 
-w /path/to/wvl.mat:wavelengths /path/to/image.mat:img_dat

will load the ref_img variable from ref.mat as the reference image,
the wavelengths variable from the wvl.mat file as the image
wavelengths, and the img_dat variable from image.mat as the image data
file.

## Credit
Based on code by Daniel Kornhauser from the matplotlib-users mailing 
list available at the following url:
http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg00662.html
