function seqs_struct = convert_to_struct(time_filename, mark_filename)

seqs_struct = struct();
time_vect = load(time_filename);
mark_vect = load(mark_filename);
seqs_struct(1).Time = time_vect.time;
seqs_struct(1).Mark = mark_vect.mark;
seqs_struct(1).Start = 0;
seqs_struct(1).Stop = max(time_vect.time); % maximum timestamp over all units is 545