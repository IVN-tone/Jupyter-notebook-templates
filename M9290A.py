import numpy as np
import visa
import os
from qsweepy.instrument import Instrument

current_channel = 1
# power_dBm = -20
# v1.write("SOURce{0}:POWer {1}".format(current_channel, power_dBm))
# v1.query("SOURce%d:POWer?"%current_channel)


class M9290A(Instrument):
	def __init__(self, name, address):
		Instrument.__init__(self, 'M9290A', tags=['physical'])
		self._address = address # see above
		self.current_channel = 1
		self._visainstrument = visa.ResourceManager().open_resource(self._address)
		
	def set_res_bw(self, res_bw): # set resolution bw of SA
		self._visainstrument.write('SENS%i:BWID:RES %i' % (current_channel, res_bw))

	def set_video_bw(self, video_bw): # set video bw of SA
		self._visainstrument.write('SENS%i:BWID:VID %i' % (current_channel, video_bw))

	def set_center_frequency(self, frequency): # set center frequency of SA
		self._visainstrument.write('SENS%i:FREQ:CENT %i' % (current_channel, frequency))

	def set_tr_avg(self, trace):
		self._visainstrument.write('AVER:TYPE RMS')
		self._visainstrument.write('DET:TRAC%i AVER'%(trace))
    
	def set_sweep_time(self, time): # set sweep time of SA
		self._visainstrument.write(':SENS{:d}:SWE:TIME {:e}'.format(current_channel, time))

	def set_span(self, span): # set center frequency span of SA
		self._visainstrument.write('SENS%i:FREQ:SPAN %i' % (current_channel, span))

	def set_nop(self, nop): # set number of points in scan of SA
		self._visainstrument.write('SENS%i:SWE:POIN %i' % (current_channel, nop))	
	
	def set_data_format(self):
		self._visainstrument.write(':FORMAT REAL,32; FORMat:BORDer SWAP;')

	def set_common_commands(self):
		self._visainstrument.write('*ESE 1')
		self._visainstrument.write('*CLS')
		self._visainstrument.write('*OPC')
		
	def get_res_bw(self): # get resolution bw of SA
		return float(self._visainstrument.query('SENS:BWID:RES?'))

	def get_video_bw(self): # get video bw of SA
		return float(self._visainstrument.query('SENS:BWID:VID?'))

	def get_center_frequency(self): # get center frequency of SA
		return float(self._visainstrument.query('SENS:FREQ:CENT?'))
	
	def get_sweep_time(self): # get sweep time of SA
		return float(self._visainstrument.query(':SENS:SWE:TIME?'))

	def get_span(self): # get center frequency span of SA
		return float(self._visainstrument.query('SENS:FREQ:SPAN?'))

	def get_nop(self): # get number of points in scan of SA
		return int(self._visainstrument.query('SENS:SWE:POIN?'))
    
	def get_data_format(self):
		return self._visainstrument.query("FORMAT:TRACE:DATA?; :FORMAT:BORDER?")
	
	def get_trace_data(self):
		data = self._visainstrument.query_binary_values("CALCulate:DATA?",datatype=u'f')
		data_size = np.size(data)
		datax = np.array(data[0:data_size:2])
		datay = np.array(data[1:data_size:2])
		return [datax, datay]