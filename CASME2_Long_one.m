clc;
clear;
FPS = 200;
Interval = 40;   %%%% 0.2 0.3 0.4
PATH = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2_Sliding_40\CASME2_Long\';
Path   = 'F:\Micro_Expression_Spotting_GLASTCNN\CASME2_Related\CASME2\';
Subject = dir(Path);
for i = 3:length(Subject)
     Sequence = dir(strcat(Path,Subject(i).name));
     for j = 3:length(Sequence)
          Image = dir(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','*.jpg'));
          
          %%% 如果图像的帧数小于或等于100
          if  length(Image) <= (FPS/2)
              if   ~exist(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_','1_100'))
                    mkdir(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_','1_100'));
              end
              for x = 1:length(Image)
                   I = imread(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','img',num2str(x),'.jpg'));
                   if length(num2str(x)) == 1
                      newname_x = strcat('000',num2str(x),'.jpg');
                   elseif length(num2str(x)) == 2
                      newname_x = strcat('00',num2str(x),'.jpg');
                   elseif length(num2str(x)) == 3
                      newname_x = strcat('0',num2str(x),'.jpg');
                   else
                      newname_x = strcat(num2str(x),'.jpg');
                   end
                   imwrite(I,strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_','1_100','\',newname_x));
              end
              
              Last_I = imread(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','img',num2str(length(Image)),'.jpg'));
              for x1 = (length(Image)+1) : (FPS/2)
                   if length(num2str(x1)) == 1
                      newname_x1 = strcat('000',num2str(x1),'.jpg');
                   elseif length(num2str(x1)) == 2
                      newname_x1 = strcat('00',num2str(x1),'.jpg');
                   elseif length(num2str(x1)) == 3
                      newname_x1 = strcat('0',num2str(x1),'.jpg');
                   else
                      newname_x1 = strcat(num2str(x1),'.jpg');
                   end
                   imwrite(Last_I,strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_','1_100','\',newname_x1));
              end 
          end
           
          %%%--------------------------------------------------------如果图像的帧数大于100 
          if  length(Image) > (FPS/2)
              N = floor((length(Image) - (FPS/2))/(Interval)) +1;
              for y = 1:N
                   sw_start = (y-1)*Interval+1;
                   sw_end = (y-1)*Interval+100;
                   name = strcat(num2str(sw_start),'_',num2str(sw_end));
                   if   ~exist(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name))
                          mkdir(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name));
                   end
                   for z = sw_start : sw_end
                        if length(num2str(z)) == 1
                               newimage_z = strcat('000',num2str(z),'.jpg');
                        elseif length(num2str(z)) == 2
                               newimage_z = strcat('00',num2str(z),'.jpg');
                        elseif length(num2str(z)) == 3
                               newimage_z = strcat('0',num2str(z),'.jpg');
                        else 
                               newimage_z = strcat(num2str(z),'.jpg');
                        end
                        II = imread(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','img',num2str(z),'.jpg'));
                        imwrite(II,strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name,'\',newimage_z));
                   end
              end
              
              sw_start1 = N*Interval+1;
              sw_end1 = N*Interval +100;
              name1 = strcat(num2str(sw_start1),'_',num2str(sw_end1));
              if   ~exist(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name1))
                     mkdir(strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name1));
              end 
              for z1 = sw_start1:length(Image)
                   III = imread(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','img',num2str(z1),'.jpg'));
                   if length(num2str(z1)) == 1
                            newimage_z1 = strcat('000',num2str(z1),'.jpg');
                   elseif length(num2str(z1)) == 2
                            newimage_z1 = strcat('00',num2str(z1),'.jpg');
                   elseif length(num2str(z1)) == 3
                            newimage_z1 = strcat('0',num2str(z1),'.jpg');
                   else 
                            newimage_z1 = strcat(num2str(z1),'.jpg');
                   end
                   imwrite(III,strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name1,'\',newimage_z1));
              end
              
               Last_III = imread(strcat(Path,Subject(i).name,'\',Sequence(j).name,'\','img',num2str(length(Image)),'.jpg'));
               for z2 = (length(Image)+1) : sw_end1
                    if length(num2str(z2)) == 1
                       newimage_z2 = strcat('000',num2str(z2),'.jpg');
                    elseif length(num2str(z2)) == 2
                       newimage_z2 = strcat('00',num2str(z2),'.jpg');
                    elseif length(num2str(z2)) == 3
                       newimage_z2 = strcat('0',num2str(z2),'.jpg');
                    else
                       newimage_z2 = strcat(num2str(z2),'.jpg');
                    end
                   imwrite(Last_III,strcat(PATH,Subject(i).name,'\',Sequence(j).name,'_',name1,'\',newimage_z2));
               end
         end
          %%%-----------------------------------------------------------------针对某个视频序列
     end
end