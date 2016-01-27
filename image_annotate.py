#!/usr/bin/env python
"""
Simple interactive image annotation tool.

Mouse commands:
  left click:    define new ROI point, drag for free-hand drawing, 
                 select/deselect existing ROI
                 
  right click:   close current ROI polygon, delete existing ROI
    
Keyboard commands:
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
"""

import matplotlib

import pylab as pl
import numpy as np

import getopt, sys, os
from os.path import join as pathjoin
from os.path import exists as pathexists

from matplotlib.nxutils import points_inside_poly
from matplotlib.patches import Rectangle


help_msg = '''Mouse commands:
  left click:    define new ROI point, drag for free-hand drawing, 
                 select/deselect existing ROI
                 
  right click:   close current ROI polygon, delete existing ROI
    
Keyboard commands:
  h: print this help message
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
'''
    
usage_msg = '''Usage:
python image_annotate.py [options] [image_file]

[image_file] can be:
  - [filename].[ext]:     any image readable by matplotlib.imread
  - [filename].[img|hdr]: an ENVI image
  - [filename].mat:[var]: a Matlab .mat file containing variable [var]
  
options:
  -h  --help         print this usage and exit
  -r  --ref          reference image file 
  -b  --bands        indices of reference bands (either 1 or 3 comma-delimted values, default=[0])
  -w  --wvl          wavelength file
  -l  --load-rois    load ROIs from file
  -c  --color        color file (default=cm_jet16.txt)
  -n  --norm         L2 normalize each pixel in image_file
  -s  --scale        scaling factor for reference image values (default=1.0)
  -f  --flip         flip data up/down w.r.t. reference file
  -v  --verbose      enable verbose output'''  



def load_matlab(filename,keys):
    mat = loadmat(filename)
    dat = []
    if type(keys)==str: 
        keys = [keys]
    for key in keys:            
        if key not in mat:
            print '%s not present in %s, skipping.'%(key,filename)
            continue            
        dat.append(mat[key])         
        
    if len(dat) == 1: # singleton
      return dat[0]

    return dat 

class ROI:
    def __init__(self, id, verts, label=' ', group=0):
        self.id = id
        self.group = group
        self.verts = verts
        self.label = label
        self.nverts = len(verts)
        self.selected = False
        self.updateXY()

    def __getitem__(self, i):
        return self.verts[i]

    def __str__(self):
        return 'ROI %d label="%s" group=%d'%(self.id,self.label,self.group)

    def __len__(self):
        return self.nverts

    def __iter__(self):
        return iter(verts)

    def updateXY(self):
        self.x = [v[0] for v in self.verts]
        self.y = [v[1] for v in self.verts]

    def inside(self, points):
        return points_inside_poly(points, self.verts)

    def center(self):
        mnx,mxx = np.min(self.x),np.max(self.x)
        mny,mxy = np.min(self.y),np.max(self.y)
        return ((mxx-mnx)/2),((mxy-mny)/2)

    def move(self,dx,dy):
        for i in range(len(self)):
            self.verts[i][0]+=cx
            self.verts[i][1]+=cy
        self.updateXY()        

class LABELER:    
    def __init__(self, img, ref, wvl=None, colortab=None,
                 roifile='rois.mat', verbose=False):

        self.img = img
        self.ref = ref

        self.rois = []
        self.roifile = roifile
            
        # polygon point bookkeeping
        self.previous_point = []
        self.start_point = []
        self.line = None
        
        self.points = []
        self.exported = True

        self.wvl = wvl 
        if self.wvl is None:
            self.wvl = range(img.shape[2])
        self.wvlmin = min(self.wvl)
        self.wvlmax = max(self.wvl)

        self.imgmin = 0.0 #np.min(self.img)
        self.imgmax = -1.0 #np.max(self.img)

        if colortab is None: # pick colors randomly
            if verbose:
                print "Using default (16-color) colortable"
            # np.random.seed(2)
            # colortab = np.random.random([16,3])
            # colortab = np.c_[colortab,np.ones([16,1])*0.75]            
            colortab = [pl.cm.hsv(i/16.0) for i in range(16)]
            
        ncolors = len(colortab)
        self.colorfn = lambda idx: colortab[idx%(ncolors-1)]
                
        self.yinch = 11
        self.dpi   = self.img.shape[0]/self.yinch
        self.xinch = self.img.shape[1]/float(self.dpi)

        if verbose:
            print "xinch=%f, yinch=%f, dpi=%f"%(self.xinch,self.yinch,self.dpi)
    
        self.zoomsz = 30
        self.mboxidx = -1

        # image/polygon figure
        self.fig = pl.figure(figsize=(self.xinch,self.yinch))        
        self.imax = self.fig.add_subplot(111, autoscale_on=False) # image axis, don't clear        
        self.imax.imshow(self.ref)
        self.fig.subplots_adjust(left=0.05,right=0.95,bottom=0.0,top=1.0,wspace=0.0)

        # zoom/stats/spectrum plot figure
        self.sfig = pl.figure(figsize=(7,5))
        self.zmax = self.sfig.add_subplot(221,xlim=(0.4,0.6),ylim=(0.4,0.6),
                                          autoscale_on=False)
        self.zmax.imshow(self.ref)
        
        self.stax = self.sfig.add_subplot(222)
        self.spax = self.sfig.add_subplot(212)

        self.stax.add_patch(Rectangle((0,0),1,1,fill=False,clip_on=False))
        
        self.sfig.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.9)

        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        self.sfig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        self.set_axis_limits()        

        if roifile is not None:
            self.restore_rois(self.roifile)
            self.draw_rois()
            
        self.update_sfig(self.zoomsz/2,self.zoomsz/2)

    def get_ids(self):
        return [roi.id for roi in self.rois]

    def get_groups(self):
        return [roi.group for roi in self.rois]

    def get_labels(self):
        return [roi.label for roi in self.rois]

    def get_selected(self):
        selected = []
        for i,roi in enumerate(self.rois):
            if roi.selected:
                selected.append(i)
        return selected

    def new_id(self):
        '''
        Generates a new ROI id sequentially from the current set of ROIs
        '''
        ids = sorted(self.get_ids())
        newid = -1
        for newid in range(len(ids)):
            if newid+1 not in ids:
                break
        return newid+1

    def new_group(self):
        '''
        Generates a new group id sequentially from the current set of ROIs
        '''        
        groups = np.unique(self.get_groups())
        newgroup = -1
        for newgroup in range(len(groups)):
            if newgroup+1 not in groups:
                break
        return newgroup+1
    
    def update_sfig(self, x, y):
        if ( x > self.img.shape[1] ) or ( y > self.img.shape[0] ):
            return
        self.update_pixel(x,y)
        self.update_zoom(x,y)
        self.update_stats(x,y)
        self.sfig.canvas.draw()

    def update_pixel(self, x, y):
        self.spax.clear()
        pixel = self.img[y,x]
        self.spax.plot(self.wvl,pixel)
        self.spax.set_xlim(self.wvlmin,self.wvlmax)
        if self.imgmax != -1.0:
            self.spax.set_ylim(self.imgmin,self.imgmax)

    def update_zoom(self, x, y):
        zhalf = self.zoomsz/2.0
        self.mnx,self.mxx = max(x-zhalf,0),min(x+zhalf,self.ref.shape[1]-1)
        self.mny,self.mxy = max(y-zhalf,0),min(y+zhalf,self.ref.shape[0]-1)
        self.zmax.set_xlim(self.mnx,self.mxx)
        self.zmax.set_ylim(self.mny,self.mxy)
        self.zmax.set_xticks([int(self.mnx),int(x),int(self.mxx)])
        self.zmax.set_yticks([int(self.mny),int(y),int(self.mxy)])
        xpts = [self.mnx,self.mnx,self.mxx,self.mxx]
        ypts = [self.mny,self.mxy,self.mxy,self.mny]
        
        if self.mboxidx != -1 and len(self.imax.patches)>0:
            del self.imax.patches[self.mboxidx]        
        self.imax.fill(xpts,ypts,fc='none',lw=1,ec='r')        
        self.mboxidx = len(self.imax.patches)-1

    def update_stats(self, x, y):
        global curpos

        self.stax.texts = []

        l, w = 0.05, 0.9
        b, h = 0.05, 0.9
        r, t = (l+w), (b+h)
        curpos,fontsz = t, 0.085

        def println(txt):
            global curpos
            self.stax.text(l,curpos,txt,verticalalignment='top')
            curpos -= fontsz

        println('Cursor position (%d,%d)'%(x,y))
        roilab,roiidx = 'None','None'
        inside = self.inside_rois(x,y)
        if inside: 
            roiidx = '%d'%max(inside)
            roilab = self.rois[max(inside)].label
            
        println('roi: %s'%roiidx)
        println('roi label: %s'%roilab)
        
        self.stax.set_axis_off()

    def inside_rois(self, x, y):
        '''
        Returns the set of ROIs which contain (x,y)
        '''
        inside = [] 
        for i,roi in enumerate(self.rois):  
            if roi.inside([[x,y]])[0]: 
                inside.append(i)
        return inside

    def draw_rois(self):
        self.mboxidx = -1
        self.imax.patches = []
        self.imax.lines = []
        self.imax.texts = []
        for i,roi in enumerate(self.rois):
            xpts,ypts = roi.x,roi.y
            roicolor = self.colorfn(roi.group)            
            if roi.selected:
                self.imax.fill(xpts,ypts,fc=roicolor,lw=2,ec='k',ls='dashed')
            else:
                edgecolor = self.colorfn(roi.id)
                self.imax.fill(xpts,ypts,color=roicolor,lw=1,ec=edgecolor)                
            self.imax.text(xpts[0],ypts[0],'%d.%d'%(roi.group,roi.id),
                           bbox=dict(facecolor='white', alpha=0.5))

        self.draw_legend(self.imax)
        self.fig.canvas.draw()

    def label_groups(self):
        gmap = {}
        for groupi in np.unique(self.get_groups()):
            grouplab = raw_input('Enter label for group %d: '%groupi)
            if len(grouplab) > 0:
                gmap[groupi] = grouplab

        for i in range(len(self.rois)):
            groupi = self.rois[i].group
            if groupi in gmap:
                self.rois[i].label = gmap[groupi]

        self.exported = False

    def draw_legend(self, ax):
        glabmap = dict([(roi.group,roi.label) for roi in self.rois])
        gpatches,glabels = [],[]
        for gid in sorted(glabmap.keys()):
            glabel='unlabeled' if gid == 0 else glabmap[gid]            
            gpatches.append(Rectangle((0,0),1,1,fc=self.colorfn(gid)))
            glabels.append('%2.0f: %s'%(gid,glabel))

        if len(gpatches) != 0:
            ax.legend(gpatches,glabels,loc=1)            
        
    def restore_rois(self,roifile):     
        from scipy.io import loadmat
        if not os.path.exists(roifile):
            return           
        roidat = loadmat(roifile)
        ids = roidat['ids'].squeeze()
        if len(ids)==0:
            return
        print 'Restoring %d rois'%len(ids)
        groups = roidat['groups'].squeeze()
        verts = roidat['verts'].squeeze()
        labels = roidat['labels'].squeeze()
        for i,id in enumerate(ids):
            label = labels[i]
            group = groups[i]            
            if len(label)>1:
                label = label.strip() 
            roi = ROI(id,verts[i],label=label,group=group)
            self.rois.append(roi)
        print 'Finished restoring rois'
        
    def export_rois(self):
        from scipy.io import savemat
        print 'Exporting %d rois to %s'%(len(self.rois),self.roifile)

        idmask,groupmask = self.get_masks()
            
        for i in np.unique(groupmask)[1:]:
            print '%d pixels in group %d'%(np.sum(groupmask==i),i)
            print '%d rois in group %d'%(len(np.unique(idmask[groupmask==i])))
            
        ids = [roi.id for roi in self.rois]
        groups = [roi.group for roi in self.rois]
        labels = [roi.label for roi in self.rois]
        verts = [roi.verts for roi in self.rois]
        
        outdict={'idmask':idmask,'groupmask':groupmask,
                 'verts':verts,'ids':ids,'labels':labels,
                 'groups':groups}
        savemat(self.roifile,outdict)
        self.exported = True
        print 'Finished exporting rois'

    def get_masks(self):
        ny,nx = self.img.shape[:2]
        x,y = np.meshgrid(np.arange(nx), np.arange(ny))
        x,y = x.flatten(), y.flatten()        
        points = np.vstack((x,y)).T
        idmask = np.zeros([ny,nx],int)
        groupmask = np.zeros([ny,nx],int)

        for i,roi in enumerate(self.rois):
            maski = roi.inside(points).reshape((ny,nx))
            idmask[maski] = roi.id
            groupmask[maski] = roi.group

        return idmask, groupmask

    def export_image(self):
        self.meanfig = pl.figure(figsize=(self.xinch,self.yinch))
        self.meanax = self.meanfig.add_subplot(111)
        self.meanfig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        
        idmask,groupmask = self.get_masks()

        shift,space = 0.0,0.1
        for gid in np.unique(self.get_groups()):
            gpix = self.img[groupmask==gid]
            gmean = np.mean(gpix,axis=0)+shift
            gstd = np.std(gpix,axis=0)
            gmin,gmax = gmean-gstd,gmean+gstd
            self.meanax.plot(self.wvl,gmean,color=self.colorfn(gid))
            self.meanax.plot(self.wvl,gmin,':',color=self.colorfn(gid))
            self.meanax.plot(self.wvl,gmax,':',color=self.colorfn(gid))

            shift = np.max(gmax)+space

        self.meanax.set_yticks([])
        self.meanax.set_xlim(self.wvlmin,self.wvlmax)
        self.draw_legend(self.meanax)
        self.meanfig.canvas.draw()
        self.meanfig.show()
            
    def zoom(self,axes,io):
        if axes in (self.imax,self.zmax):
            if io=='+':                
                self.zoomsz = min(100,int(self.zoomsz*1.1))
            elif io=='-':
                self.zoomsz = max(6,int(self.zoomsz*0.9))
        else:
            if io=='+':                
                self.imgmax += 1
            elif io=='-':
                self.imgmax = max(1,self.imgmax-1)            

    def clear_unlabeled(self):
        toclear = []
        for i,roi in enumerate(self.rois):
            if roi.group == 0:
                toclear.append(i)
        for idx in toclear[::-1]: # delete from the end first
            del self.rois[idx]

    def group_selected(self):
        group_id = self.new_group() 
        for i in self.get_selected():
            self.rois[i].group = group_id
            self.rois[i].selected = False

    def ungroup_selected(self):
        for i in self.get_selected():
            self.rois[i].group = 0
            self.rois[i].selected = False

    def join_selected(self):
        from numpy import unique
        selected = self.get_selected()
        selected_groups = []
        selected_labels = []
        for i in selected:
            groupi = self.rois[i].group
            if groupi != 0:
                selected_groups.append(groupi)
                selected_labels.append(self.rois[i].label)
                
        if len(unique(selected_groups)) == 1:
            for i in selected:
                self.rois[i].group = selected_groups[0]
                self.rois[i].label = selected_labels[0]
                self.rois[i].selected = False
        else:
            print 'Too many groups ('+str(selected_groups)+') selected'	

    def button_press_callback(self, event):
        if event.inaxes: 
            x,y = event.xdata, event.ydata
            if event.button == 1:  # left button
                inside = self.inside_rois(x,y)
                if inside:
                    idx = max(inside)
                    self.rois[idx].selected = not self.rois[idx].selected
                    self.draw_rois()
                    self.previous_point = [x,y]
                else:    
                    self.points.append([x,y])
                    if self.line == None: # if there is no line, create a line
                        self.line = pl.Line2D([x,x], [y,y], marker = 'o')
                        self.start_point = [x,y]
                        self.previous_point =  self.start_point 
                        self.imax.add_line(self.line)
                    else: # if there is a line, create a segment
                        self.line = pl.Line2D([self.previous_point[0],x], 
                                              [self.previous_point[1],y],marker = 'o')

                        self.previous_point = [x,y]
                        self.imax.add_line(self.line)                    
            
            elif event.button == 3: # right button
                if self.line != None:
                    if len(self.points) > 2:
                        # close the polygon loop
                        self.line.set_data([self.previous_point[0], self.start_point[0]],
                                           [self.previous_point[1], self.start_point[1]])                                       
                        self.imax.add_line(self.line)

                        roi=ROI(self.new_id(),self.points)
                        self.rois.append(roi)
                        self.exported = False

                    # clear points 
                    self.points = []
                    self.line = None
                    self.draw_rois()

                elif self.line is None:
                    # delete the top roi
                    inside = self.inside_rois(x,y)
                    if inside:
                        del self.rois[max(inside)] 
                        self.exported = False
                        self.draw_rois()
                        
            self.fig.canvas.draw()
            
    def set_axis_limits(self):   
        self.imax.set_xlim(0,self.img.shape[1]-1)
        self.imax.set_ylim(0,self.img.shape[0]-1)        
        
    def motion_notify_callback(self, event):
        if event.inaxes: 
			x,y = event.xdata, event.ydata
			if event.button == None and self.line != None:
				# Move line around 
				self.line.set_data([self.previous_point[0],x],
                                   [self.previous_point[1],y])      

			elif event.button == 1:
                # Free hand drawing
				line = pl.Line2D([self.previous_point[0],x],
                                 [self.previous_point[1],y])                  
				self.points.append([x,y])
				self.imax.add_line(line)
				self.previous_point = [x,y]
			elif event.button == 3:
				pass
			self.update_sfig(x,y)
			self.fig.canvas.draw()

    def button_release_callback(self, event):
        if event.inaxes:
            x,y = event.xdata, event.ydata
            if event.button == 1:
                if self.line is None and self.points:
                    # close free hand drawing
                    self.points.append([x,y])
                    self.previous_point = [x,y]
                    print 'release_previous'
                
    def key_press_callback(self, event):
        if event.key == 'h':
            print help_msg
        elif event.key == 'e': # export
            self.export_rois()
        elif event.key == 'm': # export mean image
            self.export_image()
        elif event.key == 'a': # annotate
            self.label_groups()
            self.draw_rois()
        elif event.key == 'd': # delete last roi
            del self.rois[-1]
            self.draw_rois()
        elif event.key == 'g': # group selected
            self.group_selected()
            self.draw_rois()
        elif event.key == 'u': # ungroup selected
            self.ungroup_selected()
            self.draw_rois()
        elif event.key == 'j': # join unlabeled to labeled
            self.join_selected()
            self.draw_rois()
        elif event.key == 'c': # clear unlabeled
            self.clear_unlabeled()
            self.draw_rois()
        elif event.key == '=': # grow zoom window FIXME: shift doesn't work
            self.zoom(event.inaxes,'+')            
        elif event.key == '-': # shrink zoom window
            self.zoom(event.inaxes,'-')
        elif event.key == 'q': # quit (and export if roi set changed)
            if not self.exported:
                yn = raw_input('ROIs changed, export to file? ')
                if yn.lower() == 'y':
                    self.export_rois()
            sys.exit(1)



def main(argv=None):
    class usage(Exception):
    	def __init__(self, msg):            
            self.msg = usage_msg

    if argv is None:
        argv = sys.argv
    try:
        try:
            longopts   = ['verbose','flip','norm','help']	  
            longoptsp  = ['dat','ref','bands','matkey','load-rois','scale','wvl']   
            shortopts  = ''.join([o[0] for o in longopts])
            shortopts += ''.join([o[0]+':' for o in longoptsp])
            opts, args = getopt.getopt(argv[1:], shortopts, longopts)
            datf = args[0]
            
        except (getopt.error, IndexError), msg:
            raise usage(msg)
 
        verbose = False
        flipdat = False
        norm = False
        refbands = [0,1,2]
        scalec = 1.0
        bbox,wvlf = None,None        
        reff,refkey = None,None
        colorf,roif = None,None
        for opt, val in opts:
            if opt in ('--verbose','-v'):
                verbose=True
            elif opt in ('--help','-h'):
                print __doc__
                print usage_msg
                sys.exit(0)
            elif opt in ('--bands','-b'):
                refbands = np.asarray(val.split(','),int)
            elif opt in ('--ref','-r'):
                reff = val.split(':')
                if len(reff)==2:
                    refkey = reff[1]
                reff = reff[0]
            elif opt in ('--colorfile', '-c'):
                colorf = val
            elif opt in ('--flip', '-f'):
                flipdat = True
            elif opt in ('--norm', '-n'):
                norm = True
            elif opt in ('--load-rois', '-l'):
                roif = val
            elif opt in ('--scale','-s'):
                scalec = float(val)
            elif opt in ('--wvl', '-w'):
                wvlf = val

        if len(args) != 1:
            raise usage(usage_msg)

        print datf
    
        datf = datf.split(':')
        if len(datf)==2:
            datkey = datf[1]
        datf = datf[0]
    
        if len(refbands) > 3:
            print "Too many refbands"
            return

        colortab = None
        if colorf is not None:
            #    colorf = 'cm_jet16.txt'
            colortab = np.loadtxt(colorf,dtype=float)
            colortab = np.c_[colortab,np.ones([colortab.shape[0],1])*0.75]            
            
        dfile,dext = os.path.splitext(datf)
        normf = '%s_normed.mat'%dfile
        
        if norm and pathexists(normf):
            print "Loading precomputed normalized data file %s"%normf
            dat = load_matlab(normf,'normed')
            print "Finished loading normalized data"
                   
        if dext == '.mat':
            dat = load_matlab(datf,datkey)
            if len(dat)==0:
                return -1
        elif dext in ('.img','.hdr'):
            import spectral.io.envi as envi
            dinfo = envi.open(datf)
            dat = dinfo.read_bands(range(dinfo.shape[2])).squeeze()
        else:
            dat = pl.imread(datf)

        x,y,z = np.atleast_3d(dat).shape

        if verbose:
            print "Loaded %d x %d x %d image data from file %s"%(x,y,z,datf)

        if norm and not pathexists(normf):
            from scipy.io import savemat 
            print "Saving normalized data to file %s"%normf  
            dat = dat.reshape([x*y,z])   
            normv = np.apply_along_axis(np.linalg.norm,1,dat)
            normv[normv==0] = 1.0
            dat = (dat.T / normv.T).T            
            dat = dat.reshape([x,y,z])               
            savemat(normf,{'normed':dat})
            print "Finished saving normalized data"            

        if flipdat:
            dat = dat[::-1,:,:]
                
        if reff is None:
            ref = dat[:,:,refbands]
        else:
            rfile,rext = os.path.splitext(reff)
            if rext == '.mat':
                ref = load_matlab(reff,refkey)
                if len(ref)==0:
                    return -1
            else:
                ref = pl.imread(reff)   

        if scalec != 1.0:
            reftype = ref.dtype
            ref = (ref.astype(float)*scalec).astype(reftype)

        if verbose:
            print "Reference image bands=%s, scaling factor=%f"%(refbands,scalec)
    
        if wvlf is None:
            wvl = np.arange(dat.shape[2])
        else:
            wvl = load_matlab(wvlf,'wvl')
        
        if tuple(dat.shape[:2]) != tuple(ref.shape[:2]):
            print 'Data and Reference images not the same size:',
            print dat.shape, ref.shape
            return -1
        
        if roif is None:
            roif = dfile+'_rois.mat'

        lab = LABELER(dat,ref,wvl=wvl,colortab=colortab,roifile=roif,
                      verbose=verbose)

        pl.ioff()
        pl.show()

    except usage, err:
        print >>sys.stderr, err.msg
        return 2	

if __name__ == '__main__':
    # import psyco if available
    try:
        import psyco
        psyco.full() # optimize!
    except ImportError:
        pass

    sys.exit(main())

