clc;
clear;
PATH = '\';
path  = '';

[~,~,Label] = xlsread('\CASME2.xlsx');
[Col,~] = size(Label);
for i = 2:(Col-1)
      subject = strcat('sub',Label{i,1});
      sequence = Label{i,2};
      
      %%%%----------------------------------该序列的起始与结束
      Onset = Label{i,4};
      Offset = Label{i,6};
      Wg = reshape(Onset:Offset,1,(Offset-Onset+1));
      %%%%----------------------------------------------------------
       
      SW = dir(strcat(path,subject));
      for j = 3:length(SW)
           swname = SW(j).name;
           Judge = strfind(swname,sequence);
           [col,~] = size(Judge);
           if col>0
               
              point = strfind(swname,'_');
              sw_start = str2double(swname((point(1,2)+1):(point(1,3)-1)));
              sw_end = str2double(swname((point(1,3)+1):length(swname)));
           
              Ws = reshape(sw_start:sw_end,1,(sw_end - sw_start+1));
              Wspotted = intersect(Ws, Wg);
              
              Rate = length(Wspotted)/(Offset-Onset+1);  %%% 此窗口对[onset - offset] 的占比
              Num = 100000;
              for k = 4:8
                   if   k/10 <= Rate
                        Num = Num+10^(k-4);
                   end
              end
              Str = num2str(Num);
              Filename = Str(2:length(Str));
              if ~exist(strcat(PATH,subject,'\',swname,'_',Filename))
                   mkdir(strcat(PATH,subject,'\',swname,'_',Filename));
              end
              Image = dir(strcat(path,subject,'\',swname,'\','*.jpg'));
              for x = 1:length(Image)
                   I = imread(strcat(path,subject,'\',swname,'\',Image(x).name));
                   imwrite(I,strcat(PATH,subject,'\',swname,'_',Filename,'\',Image(x).name));
              end
          end
     end
end