# About
In 2023 I took a course taught by Professor Qi Guo at Purdue University in machine learning and statistics. The final project was to reimplment a seminal machine learning publication. I elected to re-implement the paper <i>Plug and Play ADMM for Image Restoration</i> by Dr. Stanley Chan. My work earned me a 	
86.0 / 84.0 on the final project and an A in the class. This GitHub Repository is my work on this final project as well as the paper.


# Differences with original code base
Since this is a re-implmentation of a very well documented project, there is an existing codebase for the original project [here:](https://www.mathworks.com/matlabcentral/fileexchange/60641-plug-and-play-admm-for-image-restoration). This github does three things differently than this original codebase:
<ol>
<li> That codebase is in MatLab while this one is in Python3
<li> The parameters are a bit different - not better or worse but it's a re-implmementation in a different language so the values are different
<li> This project expands upon the old one and tests results for different scenarios than the original which only tested on a few
<li> The PNSR is lower in general for this project because the noise is higher than in the MatLab demo
</ol>

# How to use
## Demo
To see the ADMM algorithm in action, go to the `demonstrations` folder. All scripts can be run with python and will display the before and after denoised image. You should run the files from within the demonstrations folder.

## Analysis
In the final project document, there are graphs displaying how the Peak to Signal Noise Ratio (PNSR) changes given different parameters. This is calculated with the scripts in the `analysis` folder. These scripts take a long time, sometimes several days, to run. I highly reccomend using other computing recourses. Also, the files can load python pickles from the pickles folder if you don't want to generate the data from scratch. The generated plots are also in the `analysis` folder.
