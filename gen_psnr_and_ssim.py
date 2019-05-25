import os
import numpy
import scipy.misc
from psnr import psnr
from ssim import ssim_exact
from glob import glob


# ref_fileA = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\input.png'
# refA = scipy.misc.imread(ref_fileA).astype(numpy.float32)
# print(numpy.shape(refA))

# ref_filePRS = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\proposed.png'
# refPRS = scipy.misc.imread(ref_filePRS).astype(numpy.float32)

# ref_fileMoodoki = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\raymond.png'
# refMoodoki = scipy.misc.imread(ref_fileMoodoki).astype(numpy.float32)

# _psnrPRS = psnr(refA, refPRS)
# _psnrMoodoki = psnr(refA, refMoodoki)
# print([_psnrPRS, _psnrMoodoki])

# print([ssim_exact(refA, refPRS), ssim_exact(refA, refMoodoki)])

inDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\input'
imgExt = 'png'
outDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\output_center_mask'
#outDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\output_random_mask'
imgSize = [64, 64]

psnrs = []
ssims = []
imgfilenames = glob(inDir + '/*.' + imgExt)
len_ = len(imgfilenames)
i = 0
for imgfilename in imgfilenames:
    lastIndex = imgfilename.rfind("\\")
    fileName = imgfilename[lastIndex+1:]
    fileName = fileName[fileName.index('_')+1:]

    #  ref_img = scipy.misc.imresize(scipy.misc.imread(
    #     imgfilename, flatten=True), size=imgSize, interp='cubic')

    ref_img = imgfilename
    ref_img = scipy.misc.imread(ref_img, flatten=True)

    ref_prsFile = outDir+'\\prs\\inpaint_'+fileName
    ref_prsFile = scipy.misc.imread(ref_prsFile, flatten=True)
    ref_moodokiFile = outDir + '\\moodoki\\inpaint_' + fileName
    ref_moodokiFile = scipy.misc.imread(ref_moodokiFile, flatten=True)

    _psnrPRS = psnr(ref_img, ref_prsFile)
    _psnrMoodoki = psnr(ref_img, ref_moodokiFile)

    tuple_ = [fileName, _psnrPRS, _psnrMoodoki]
    psnrs.append(tuple_)
    # print(tuple_)

    _ssimPRS = ssim_exact(ref_img/255.0, ref_prsFile/255.0)
    _ssimMoodoki = ssim_exact(ref_img/255.0, ref_moodokiFile/255.0)

    tuple_ = [fileName, _ssimPRS, _ssimMoodoki]
    ssims.append(tuple_)
    # print(tuple_)
    # break
    i = i + 1
    print('%d/%d \r' % (i, len_), end='')

    # refPRS = scipy.misc.imread(imgfilename

psnrPrs = []
psnrMoodoki = []
with open('prsn_results.csv', mode='wt', encoding='utf-8') as f:
    f.write('Frame ID, Proposed PSRN, Raymond PSNR\n')
    for i in range(len(psnrs)):
        value = psnrs[i]
        psnrPrs.append(value[1])
        psnrMoodoki.append(value[2])
        f.write('%s,%f,%f' % (value[0], value[1], value[2])+'\n')

ssimPrs = []
ssimMoodoki = []
with open('ssims_results.csv', mode='wt', encoding='utf-8') as f:
    f.write('Frame ID, Proposed SSIM, Raymond SSIM\n')
    for i in range(len(ssims)):
        value = ssims[i]
        ssimPrs.append(value[1])
        ssimMoodoki.append(value[2])
        f.write('%s,%f,%f' % (value[0], value[1], value[2])+'\n')

psnrPrsMean = numpy.mean(psnrPrs)
psnrMoodokiMean = numpy.mean(psnrMoodoki)
print('PSNR mean \nPRS: %f\nMoodoki: %f' % (psnrPrsMean, psnrMoodokiMean))

with open('psnr_results_mean.csv', mode='wt', encoding='utf-8') as f:
    f.write('# of Frames, Proposed PSRN (mean), Raymond PSRN (mean)\n')
    f.write('%d,%f,%f' % (i, psnrPrsMean, psnrMoodokiMean)+'\n')

ssimPrsMean = numpy.mean(ssimPrs)
ssimMoodokiMean = numpy.mean(ssimMoodoki)
print('SSIM mean \nPRS: %f\nMoodoki: %f' % (ssimPrsMean, ssimMoodokiMean))

with open('ssims_results_mean.csv', mode='wt', encoding='utf-8') as f:
    f.write('# of Frames, Proposed SSIM (mean), Raymond SSIM (mean)\n')
    f.write('%d,%f,%f' % (i, ssimPrsMean, ssimMoodokiMean)+'\n')
