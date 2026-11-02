clear;
clc;
PathVal     = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2_Sliding_40\Val\';
path = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2_Sliding_40\GYH\';
area = dir(path);
for z = 3:length(area)
     subject = dir(strcat(path, area(z).name));
     for i = 3:length(subject)
          sequence = dir(strcat(path, area(z).name,'\',subject(i).name));
          for j = 3:length(sequence)
               image = dir(strcat(path, area(z).name,'\',subject(i).name,'\',sequence(j).name,'\','*.jpg'));
               for k =1:(length(image)-1)
                    I= imread(strcat(path, area(z).name,'\', subject(i).name,'\',sequence(j).name,'\',image(k).name));
                    if rem(k, 3)==1
                       if ~exist(strcat(PathVal, area(z).name,'\', subject(i).name,'\','1_', sequence(j).name))
                            mkdir(strcat(PathVal, area(z).name,'\', subject(i).name,'\','1_', sequence(j).name));
                       end
                       imwrite(I, strcat(PathVal, area(z).name,'\', subject(i).name,'\','1_', sequence(j).name,'\',image(k).name));
                    end
               end
          end
     end
end