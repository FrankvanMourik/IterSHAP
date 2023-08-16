%% clear all/ close figs
close all
clear
clc

%% default paramters
Participant = 32;
Video = 40;
Channel = 32;
Fs = 128;
Time = 63;
addpath('/MATLAB Drive/data/data_preprocessed_matlab')

%% set parameters
frameNum = 60;
nr_ext_feat = 14;

%%

for participant = 1:Participant
    fprintf('\nworking on file number %d:\n', participant);
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    
    for video=1:Video
        
        fprintf('\ncreating file participant %d,video %d:\n',participant,video);
        op1 = 'participant';
        op2 = 'video';
        filename = [op1 int2str(participant) op2  int2str(video) '.txt'];filename;
        fid = fopen( filename, 'wt' );
        %fprintf(filename);
        output = zeros(nr_ext_feat, Channel, frameNum);                
        datastart=128*3;
        datalength=8064-128*3;
      
        
        for channel = 1:32
            signal=zeros(1,8064-datastart);
            for ii =1:datalength
                signal(1,ii)=data(video,channel,ii+datastart);
            end

            % data1 is an array with 60 seconds x 128 Hz
   
            % Extract time domain features per second
            start = 1;
            for k =1:frameNum
                % data to take into account (dimensions: 1x128):
                one_sec_signal = signal(1, start:Fs*k);
                start = start + Fs;
                % Calculate Mean
                meanValue = mean(one_sec_signal);
                output(1, channel, k) = meanValue;
            
                % Calculate Median
                medianValue = median(one_sec_signal);
                output(2, channel, k) = medianValue;
            
                % Calculate Standard Deviation
                stdDeviation = std(one_sec_signal);
                output(3, channel, k) = stdDeviation;
            
                % Calculate Interquartile Range
                interquartileRange = iqr(one_sec_signal);
                output(4, channel, k) = interquartileRange;
            
                % Calculate Maximum Value
                maxValue = max(one_sec_signal);
                output(5, channel, k) = maxValue;
            
                % Calculate Minimum Value
                minValue = min(one_sec_signal);
                output(6, channel, k) = minValue;

                % Do a Fast Fourier Transform and calculate the same
                % features
                fftSignal = fft(one_sec_signal);

                % Calculate Mean
                meanValue = mean(abs(fftSignal).^2);
                output(8, channel, k) = meanValue;
            
                % Calculate Median
                medianValue = median(abs(fftSignal).^2);
                output(9, channel, k) = medianValue;
            
                % Calculate Standard Deviation
                stdDeviation = std(abs(fftSignal).^2);
                output(10, channel, k) = stdDeviation;
            
                % Calculate Interquartile Range
                interquartileRange = iqr(abs(fftSignal).^2);
                output(11, channel, k) = interquartileRange;
            
                % Calculate Maximum Value
                maxValue = max(abs(fftSignal).^2);
                output(12, channel, k) = maxValue;
            
                % Calculate Minimum Value
                minValue = min(abs(fftSignal).^2);
                output(13, channel, k) = minValue;
            end


            % Extract frequency domain features
            
            
            % Append the features

            % Save as 60 samples (1 per second) with 32x14 features

            
        end

        % Reshape data to be 60 samples by 32*14 features
        output2d = reshape(output,[Channel*nr_ext_feat,frameNum]);
        for k = 1:Channel*nr_ext_feat
            fprintf(fid, '%g,',output2d(k,:));
            fprintf(fid, '\n');
        end
       
        fclose(fid);
    end %the testcase loop
end %the file loop
