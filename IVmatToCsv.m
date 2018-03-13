clear all
clc

args = ["SampleRate","TYP","POS","DUR","CHN"];

for j=1:9
	path = '/home/xn3cr0nx/Scrivania/Datasets/Datasets - BCI Competitions + JKHH/Competition IV/IV_II/'; 
	subj = strcat('A0', num2str(j), 'T');
	loading = strcat(path, subj, '.mat');
	data = load(loading)

	path = '/home/xn3cr0nx/Scrivania/Datasets/csv/CompetitionIV/IV_II/';
	signal = strcat(path, subj, '/original_signal.csv');
	dlmwrite(signal, data.original_signal, 'delimiter',',','precision',7);

	info = strcat(path, subj, '/info/');
	label = strcat(info, 'Classlabel.csv');
	csvwrite(label, data.info.Classlabel);

	for i=1:4
		object = strcat('data.info.EVENT.', args(i));
		event = strcat(info, args(i), '.csv');
		dlmwrite(event, eval(object), 'delimiter',',','precision',7);	
	end
end

// args = {'TYPE', 'VERSION', 'FileName', 'T0', 'FILE', 'Patient', 'HeadLen', 'NS', 'SPR', 'NRec', 'SampleRate', 'FLAG', 'EVENT', 'Label', 'LeadIdCode', 'PhysDimCode', 'PhysDim', 'Filter', 'PhysMax', 'PhysMin', 'DigMax', 'DigMin', 'Transducer', 'Cal', 'Off', 'GDFTYP', 'TOffset', 'LowPass', 'HighPass', 'Notch', 'ELEC', 'Impedance', 'fZ', 'AS', 'Dur', 'REC', 'Manufacturer', 'InChanSelect', 'Calib', 'Classlabel', 'TRIG', 'ArtifactSelection', 'CHANTYP'};