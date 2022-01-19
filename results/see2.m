 clear,clc,close all;
filenames=dir('*_mask.png');
for i=1:length(filenames)
    filename=imread([num2str(i - 1) '_mask.png']);
    temp = double(filename);
    imwrite(temp, [num2str(i - 1) '_mask.png']);
end
