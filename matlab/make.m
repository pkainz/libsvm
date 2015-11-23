% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		mex libsvmread.c
		mex libsvmwrite.c
        
		% no OpenMP support
        %mex svmtrain.c ../svm.cpp svm_model_matlab.c
		%mex svmpredict.c ../svm.cpp svm_model_matlab.c
        
        % with OpenMP support
        setenv('CXXFLAGS', '-fopenmp') 
        mex -I.. -lgomp svmtrain.c ../svm.cpp svm_model_matlab.c
        mex -I.. -lgomp svmpredict.c ../svm.cpp svm_model_matlab.c
        
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		mex CFLAGS="\$CFLAGS -std=c99 " -largeArrayDims libsvmread.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
		
        % no OpenMP support
        %mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims svmtrain.c ../svm.cpp svm_model_matlab.c
		%mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims svmpredict.c ../svm.cpp svm_model_matlab.c
        
        % with OpenMP support
        mex CFLAGS="\$CFLAGS -std=c99" CXXFLAGS="\$CXXFLAGS -fopenmp" -largeArrayDims -I.. -lgomp svmtrain.c ../svm.cpp svm_model_matlab.c
        mex CFLAGS="\$CFLAGS -std=c99" CXXFLAGS="\$CXXFLAGS -fopenmp" -largeArrayDims -I.. -lgomp svmpredict.c ../svm.cpp svm_model_matlab.c
	end
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
