clear all
clc

args = ["run", "trial", "sample", "signal", "TargetCode", "ResultCode", "StimulusTime", "Feedback", "IntertrialInterval", "Active", "SourceTime", "RunActive", "Recording", "IntCompute", "Running"];
subjects = ["AA", "BB", "CC"];
sessions = [1:10];
for i=1:3
	for j=1:10
		path = '/home/xn3cr0nx/Scrivania/Datasets/Datasets - BCI Competitions + JKHH/Competition II/II_IIa/'; 
		subject = subjects(i);
		session = num2str(sessions(j)); 
		if(session == '10') 
			pre = '0';
		else
			pre = '00';
		end 
		ext = '.mat';
		stream = strcat(path, subject, pre, session, ext);
		subjsec = strcat(subject, pre, session);
		data = load(stream)
		path = '/home/xn3cr0nx/Scrivania/Datasets/csv/CompetitionII/';
		ext = '.csv';
		for k=1:15
			write = strcat(path, subjsec, '/', args(k), ext)
			object = strcat('data.', args(k));
			csvwrite(write, eval(object))
		end
	end
end
