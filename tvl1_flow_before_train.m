clc; clear; 
PATH = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2_Sliding_40\Train_of_before\';

path  = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2_Sliding_40\Train\';
area = dir(path);
for z = 3:length(area)
     subject = dir(strcat(path, area(z).name));
     for i = 3:length(subject)
          sw = dir(strcat(path, area(z).name, '\', subject(i).name));
          for j = 3:length(sw)
               image = dir(strcat(path,  area(z).name, '\', subject(i).name,'\', sw(j).name,'\','*.jpg'));
               firstframe = double(rgb2gray(imread(strcat(path,  area(z).name, '\', subject(i).name,'\', sw(j).name,'\',image(1).name))));
               for k = 2:length(image)
                    imagename = image(k).name;
                    currentframe = double(rgb2gray(imread(strcat(path,  area(z).name, '\', subject(i).name,'\', sw(j).name,'\',image(k).name))));
                    imflow=tvl1(firstframe, currentframe);
                    imflow = uint8(flowToColor(imflow));
                    if ~exist(strcat(PATH,  area(z).name, '\', subject(i).name, '\', sw(j).name))
                       mkdir(strcat(PATH,  area(z).name, '\', subject(i).name, '\', sw(j).name));
                    end
                    imwrite(imflow, strcat(PATH,  area(z).name, '\', subject(i).name, '\', sw(j).name,'\', image(k).name));
                    firstframe = currentframe;
               end
          end
    end
end