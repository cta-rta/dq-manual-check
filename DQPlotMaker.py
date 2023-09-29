from time import strftime, time
START=time()

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import sys

from astropy.table import Table
from glob import glob
from matplotlib import use
from pathlib import Path

#from ctapipe.visualization import CameraDisplay
#from ctapipe.coordinates import EngineeringCameraFrame
#from ctapipe.instrument import SubarrayDescription
#from ctapipe.containers import EventType
#from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
#from lstchain.io.io import dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key, dl1_params_tel_mon_flat_key



def getLogger(logName="DQPlotMaker",
              logLevel="DEBUG",
              logFormat="%(asctime)s %(levelname)s - %(message)s",
              logOutDir=None
              ):
    """
    Define a logger.
    
    Parameters
    ----------
    logName : str
        Logger Name.
    logLevel : str
        Logger Level.
    logFormat : str
        Logger Format.
    logOutDir : pathlib.Path
        Logger Output directory.
    
    Return
    ------
    log : logging.Logger
        Configured Logger.
    """
    log = logging.getLogger(logName)
    log.setLevel(logLevel)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
    log.propagate = False

    if logOutDir is not None:
        logFilePath = logOutDir.joinpath(f"{logName}.log") #_{strftime('%Y%m%d-%H%M%S')}
        fh = logging.FileHandler(str(logFilePath), 'w')
        fh.setLevel(logLevel)
        fh.setFormatter(logging.Formatter(logFormat))
        log.addHandler(fh)
        log.debug(f"Log to file: {logFilePath}")
    else:
        log.debug(f"Log set.")
        
    return log

def DQReadPickle(FilePath):
    """
    Read pickle and return its data. 
    The pickle contains the SAG-DQ analysis output.
    The DL1 analysis output will result in nested dict of dicts comprising 
    two parent keys: "events" and "quality_checks". 
    Inside "events" are found the output of the analysis as defined 
    in the DQ-LIB configuration.
    Inside "quality_checks" are found the IDs of warning and alarms 
    of values below the relative quality thresholds.
    """

    with open(FilePath, 'rb') as f:
        data = pickle.load(f)        
    return data



class DQPlotMaker():
    "Class to Verify available results and produce plots"
    def __init__(self, log, OutputDirectory) -> None:
        """Initialize only with Log object and OutputDirectory"""
        self.log = log
        self.OutputDirectory= OutputDirectory
        use("Agg")
        
    def setInputDirectory(self, InputDirectory):
        """
        Set the Input Directory and verify its content.
        
        Parameters
        ----------
        InputDirectory : pathlib.Path
            Path to the Input Directory that stores DQ results files.
        """
        self.InputDirectory = InputDirectory        
        return None
    
    def SetCameraGeometry(self, geometryfile):
        """Open a DL1 file and read camera geometry from it."""
        
        # DL1Files = glob(f"{self.InputDirectory}/line_1/*")
        # DL1Files.sort()
        # DL1File = DL1Files[0]
        # log.info(f"Look for Camera Geometry inside {DL1File}")
        
        # # Read useful instrument configuration information:
        # subarray_info = SubarrayDescription.from_hdf(DL1File)
        # #print(subarray_info.tels)
        
        # # Obtain camera geometry:
        # camgeom = subarray_info.tel[1].camera.geometry
        # # Transform camera geometry to the more usual
        # # "engineering camera frame":
        # camgeom = camgeom.transform_to(EngineeringCameraFrame())
        
        log.info(f"Read camera geometry from {geometryfile}")
        log.warning(f"TO DO: LOAD CAMERA GEOMETRY")
        
        return None
    
    def PickleToRowAnalysisCamera(self, FileName : str):
        """
        Receive the File Name of a .pickle file, open it and save
        appropriate fields as an astropy Row, easy to plot.
        
        Parameters
        ----------
        FileName : str
            File Name of the pickle file to open.
        
        Return
        ------
        DataRow : astropy.table.Row
            Row with relevant information from the pickle.
        """
        # Get the Line Number from file name assuming naming convention.
        log.info(f"Reading file {FileName}")
        line = int(re.search(r'_line_(.*?)_(.*?).pickle', FileName).group(1))
        batch_info= re.search(r'_line_(.*?)_(.*?).pickle', FileName).group(2).split("_")
        if str(batch_info[0]) == "camera":
            batch = int(batch_info[1])
            # Read the Data
            data = DQReadPickle(FileName)
            # Prepare the Astropy Row
            table = Table()
            table['line'] = np.array([line], dtype=int) # In this way Table length is set =1
            table['batch']= np.array([batch],dtype=int)
            #table['file'] = np.array([FileName], dtype=str) # This string is very long

            # Scalar columns
            table['_start_time'] = data['events'][0]['_start_time']
            table['_end_time'  ] = data['events'][0]['_end_time']
            table['rate'       ] = data['events'][0]['rate']
            table['cum_rate'   ] = data['events'][0]['cum_rate']
            # Vector columns
            table['camera_sum']=[data['events'][0]['camera_sum']]
            table['camera_sum_of_squares']=[data['events'][0]['camera_sum_of_squares']]
            #table['camera_pixels_charges_distributions']=[data['events'][0]['camera_pixels_charges_distributions']]
            
            #log.info(f"File: {FileName}")
            #log.info(f'DL1 time interval = [{data["events"][0]["_start_time"]}, {data["events"][0]["_end_time"]}] seconds')
            #log.info(f'DL1 event rate = {data["events"][0]["rate"]}')
            
            # Return the results as a Row (not Table) object.
            DataRow = table[0]
            return DataRow
        else:
            return None
    
    
    def PickleToRowAnalysisHillas(self, FileName : str):
        """
        Receive the File Name of a .pickle file, open it and save
        appropriate fields as an astropy Row, easy to plot.
        
        Parameters
        ----------
        FileName : str
            File Name of the pickle file to open.
        
        Return
        ------
        DataRow : astropy.table.Row
            Row with relevant information from the pickle.
        """
        # Get the Line Number from file name assuming naming convention.
        line = int(re.search(r'_line_(.*?)_(.*?).pickle', FileName).group(1))
        batch_info= re.search(r'_line_(.*?)_(.*?).pickle', FileName).group(2).split("_")
        if str(batch_info[0]) == "hillas":
            batch = int(batch_info[1])
            # Read the Data
            data = DQReadPickle(FileName)
            # Prepare the Astropy Row
            table = Table()
            table['line'] = np.array([line], dtype=int) # In this way Table length is set =1
            table['batch']= np.array([batch],dtype=int)
            #table['file'] = np.array([FileName], dtype=str) # This string is very long
            
            # Scalar columns
            table['_start_time'] = data['events'][0]['_start_time']
            table['_end_time'  ] = data['events'][0]['_end_time']
            table['rate'       ] = data['events'][0]['rate']
            table['cum_rate'   ] = data['events'][0]['cum_rate']
            
            ## Distribution columns
            table['intensity_distribution']=[data['events'][0]['intensity_distribution']]
            table['kurtosis_distribution' ]=[data['events'][0]['kurtosis_distribution']]
            table['length_distribution'   ]=[data['events'][0]['length_distribution']]
            table['phi_distribution'      ]=[data['events'][0]['phi_distribution']]
            table['psi_distribution'      ]=[data['events'][0]['psi_distribution']]
            table['r_distribution'        ]=[data['events'][0]['r_distribution']]
            table['skewness_distribution' ]=[data['events'][0]['skewness_distribution']]
            table['width_distribution'    ]=[data['events'][0]['width_distribution']]
            
            ## Sample columns
            ## Correlation columns
            
            # Return the results as a Row (not Table) object.
            DataRow = table[0]
            return DataRow
        else:
            return None
    
    def PickleToRowAggregationCamera(self, FileName : str):
        """
        Receive the File Name of a .pickle file, open it and save
        appropriate fields as an astropy Row, easy to plot.
        
        Parameters
        ----------
        FileName : str
            File Name of the pickle file to open.
        
        Return
        ------
        DataRow : astropy.table.Row
            Row with relevant information from the pickle.
        """
        # Read the Data
        data = DQReadPickle(FileName)
        # Prepare the Astropy Row
        table = Table()
        table['file'] = np.array([FileName], dtype=str) # This string is very long
        
        # Scalar columns
        table['_start_time'] = data['events'][0]['_start_time']
        table['_end_time'  ] = data['events'][0]['_end_time']
        table['rate'       ] = data['events'][0]['rate']
        table['cum_rate'   ] = data['events'][0]['cum_rate']
        # Vector columns
        table['camera_sum']=[data['events'][0]['camera_sum']]
        table['camera_avg']=[data['events'][0]['camera_avg']]
        table['camera_rms']=[data['events'][0]['camera_rms']]
        #table['camera_pixels_charges_distributions']=[data['events'][0]['camera_pixels_charges_distributions']]
        
        # Return the results as a Row (not Table) object.
        DataRow = table[0]
        return DataRow
    
    def PickleToRowAggregationHillas(self, FileName : str):
        """
        Receive the File Name of a .pickle file, open it and save
        appropriate fields as an astropy Row, easy to plot.
        
        Parameters
        ----------
        FileName : str
            File Name of the pickle file to open.
        
        Return
        ------
        DataRow : astropy.table.Row
            Row with relevant information from the pickle.
        """
        # Read the Data
        data = DQReadPickle(FileName)
        # Prepare the Astropy Row
        table = Table()
        # table['line'] = np.array([line], dtype=int) # In this way Table length is set =1
        # table['file'] = np.array(FileName, dtype=str) # This string is very long
        
        # Scalar columns
        
        # Return the results as a Row (not Table) object.
        DataRow = table[0]
        return DataRow
    
    def PlotScalarQuantity(self, DataTable, quantity, OutputDirectory):
        """
        Plot a scalar quantity on a 1D graph.
        X values are taken averaging `_start_time` and `_end_time` columns.
        Data from different DQ lines are plotted as different lines.
        
        Parameters
        ----------
        DataTable : `astropy.table.Table`
            Table that contains the data.
        quantity : str
            Name of the Column to plot.
        OutputDirectory : `pathlib.Path`
            Directory where plot is saved.
        """
        # Define Figure
        fig, ax = plt.subplots(1, figsize=(8,12), constrained_layout=True)
        
        # Get a list of the available line values
        LineValues = list(set(DataTable['line'])) # Save unique line numbers
        for line in LineValues:
            LineTable = DataTable[DataTable['line']==line]
            Times = (LineTable["_end_time"]+LineTable["_start_time"])/2.0
            TimesErr = (Times-LineTable["_start_time"],
                        LineTable["_end_time"]-Times)
            ax.errorbar(Times, LineTable[quantity], xerr=TimesErr, label=f"Line {line}",ls='', marker='o', markersize=5, capsize=3)
        
        ax.grid()
        ax.set_ylabel(quantity)
        ax.set_xlabel("Time")
        ax.legend()
        
        # Save Figure
        OutputFile = OutputDirectory.joinpath(f"{quantity}.png")
        self.log.info(f"Write {OutputFile}")
        fig.savefig(OutputFile)
        plt.close(fig)
        return None
    
    def PlotCameraQuantity(self, DataTable, quantity, OutputDirectory):
        """
        Plot a scalar quantity on a camera plot.
        
        Parameters
        ----------
        DataTable : `astropy.table.Table`
            Table that contains the data.
        quantity : str
            Name of the Column to plot.
        OutputDirectory : `pathlib.Path`
            Directory where plot is saved.
        """
        # Define Figure
        fig, ax = plt.subplots(1, figsize=(8,12), constrained_layout=True)
        
        # Use LSTchain
        #camdisplay = CameraDisplay(camgeom, dl1_images['image'][index])
        #camdisplay.add_colorbar()
        
        #tmin = np.min(dl1_images['peak_time'][index][cleanmask])
        #tmax = np.max(dl1_images['peak_time'][index][cleanmask])
        #camdisplay = CameraDisplay(camgeom, dl1_images['peak_time'][index], 
        #                       title='peak time (ns)')
        #camdisplay.set_limits_minmax(tmin, tmax)
        #camdisplay.add_colorbar()
        
        
        # Save Figure
        OutputFile = OutputDirectory.joinpath(f"{quantity}.png")
        self.log.info(f"Write {OutputFile}")
        fig.savefig(OutputFile)
        plt.close(fig)
        return None
    
    
    def PlotBinnedHistograms(self, DataTable, quantity, OutputDirectory):
        """
        Plot an histogram of binned data for every batch (i.e. every row).
        """
        
        for DataRow in DataTable:
            Line=DataRow['line']
            Batch=DataRow['batch']
            OutputName=OutputDirectory.joinpath(f"{quantity}_line_{Line}_{Batch}.png")
            self.PlotBinnedHistogram(DataRow, quantity, OutputName)
        
        return None
    
    
    def PlotBinnedHistogram(self, DataRow, quantity, OutputName):
        """
        Plot an histogram of binned data.
        
        Parameters
        ----------
        DataTable : `astropy.table.Table`
            Table that contains the data.
        quantity : str
            Name of the Column to plot.
        OutputDirectory : `pathlib.Path`
            Directory where plot is saved.
        """
        # Define Figure
        fig, ax = plt.subplots(1, figsize=(8,12), constrained_layout=True)
        
        # Bar plot
        heights = DataRow[quantity]
        x_values = range(len(heights)) #TODO: Insert correct x values
        ax.bar(x_values, heights)
        
        ax.grid()
        ax.set_ylabel("Distribution")
        ax.set_xlabel(f"{quantity} [A.U.]")
        ax.set_title(f"{OutputName.name}")
        
        # Save Figure
        self.log.info(f"Write {OutputName}")
        fig.savefig(OutputName)
        plt.close(fig)
        return None
    
    
    def CollectResults(self,
                       DQResultType : str,
                       DL1Class : str,
                       SaveTable = False,
                       SavePlot = True
                       ):
        """
        Look into one of the possible DQ Results folders (either analysis or aggregation),
        select the results for a Class of DL1 (either camera or hillas),
        recursively store relevant information in a table and make plots.
        
        DQResultType : str,
            Either `analysis` or `aggregation`
        DL1Class : str
            Either `camera` or `hillas`
        SaveTable :bool
            Choose if table has to be saved. 
        SavePlot : bool
            Choose if plots have to be saved.
        """
        
        if DQResultType not in ['analysis', 'aggregation']:
            raise ValueError(f"Not valid DQResultType={DQResultType}. Must be Either `analysis` or `aggregation`")
        if DL1Class not in ['camera', 'hillas']:
            raise ValueError(f"Not valid DL1Class={DL1Class}. Must be Either `camera` or `hillas`")
        
        # Define the Specific Input and Output Paths
        # for the Analysis/Aggregation, Camera/Hillas results from the general
        # Input and Output Directories
        DataDirectoryInput =  self.InputDirectory.joinpath(f"{DQResultType}")
        DataDirectoryOutput= self.OutputDirectory.joinpath(f"dq_{DQResultType}_plots/{DL1Class}")
        DataDirectoryOutput.mkdir(exist_ok=True, parents=True)
        log.info(f"Looking into {DataDirectoryInput}")
        
        # Get the File Names of the available pickles
        FileList = glob(f"{DataDirectoryInput}/*.pickle")
        FileList.sort()
        log.debug(f"Number of .pickle found: {len(FileList)}")
        
        # Build the Table with .pickle data row by row
        # Each row is read from a different pickle file
        if DQResultType=='analysis' and DL1Class=='camera':
            DataRows = [self.PickleToRowAnalysisCamera(file) for file in FileList if self.PickleToRowAnalysisCamera(file) is not None]
        elif DQResultType=='analysis' and DL1Class=='hillas':
            DataRows = [self.PickleToRowAnalysisHillas(file) for file in FileList if self.PickleToRowAnalysisHillas(file) is not None]
        elif DQResultType=='aggregation' and DL1Class=='camera':
            DataRows = [self.PickleToRowAggregationCamera(file) for file in FileList if self.PickleToRowAggregationCamera(file) is not None]
        elif DQResultType=='aggregation' and DL1Class=='hillas':
            DataRows = [self.PickleToRowAggregationHillas(file) for file in FileList if self.PickleToRowAggregationHillas(file) is not None]
        
        # Join all the Rows in a single Table
        DataTable = Table(rows=DataRows, names=DataRows[0].keys(),
                          meta={'DQResultType':DQResultType,'DL1Class':DL1Class}
                          )
        if SaveTable:
            log.info(f"Write {DQResultType} {DL1Class} table.ecsv")
            DataTable.write(DataDirectoryOutput.joinpath("table.ecsv"), overwrite=True)

        if SavePlot:
            if DQResultType=='analysis' and DL1Class=='camera':
                self.PlotScalarQuantity(DataTable=DataTable, quantity="rate", OutputDirectory=DataDirectoryOutput)
                self.PlotScalarQuantity(DataTable=DataTable, quantity="cum_rate", OutputDirectory=DataDirectoryOutput)
                #self.PlotCameraQuantity(DataTable=DataTable, quantity="camera_sum", OutputDirectory=DataDirectoryOutput)
            elif DQResultType=='analysis' and DL1Class=='hillas':
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="intensity_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="kurtosis_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="length_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="phi_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="psi_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="r_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="skewness_distribution", OutputDirectory=DataDirectoryOutput)
                self.PlotBinnedHistograms(DataTable=DataTable, quantity="width_distribution", OutputDirectory=DataDirectoryOutput)
            elif DQResultType=='aggregation' and DL1Class=='camera':
                log.error("Aggregation camera plots not defined yet.")
            elif DQResultType=='aggregation' and DL1Class=='hillas':
                log.error("Aggregation hillas plots not defined yet.")
        return None


if __name__=="__main__":
    
    ImportTime = time()-START
    
    # Define Input Directory and Check its existence
    parser = argparse.ArgumentParser(description='Collect data from multiple NPY outputs')
    parser.add_argument('--inputdirectory', '-i',  type=str, required=True, help='Directory with DQ output')
    parser.add_argument('--outputdirectory', '-o',  type=str, help='Directory with output plots')
    parser.add_argument('--camerafile', '-c', type=str, help='Path to a DL1 file containing the appropriate camera geometry')
    args = parser.parse_args()
    
    InputDirectory = Path(args.inputdirectory).absolute()
    if not InputDirectory.is_dir():
        raise NotADirectoryError(f"Not a Directory: {InputDirectory}")
    
    # Define Output Directory and Logger
    if args.outputdirectory is None:
        OutputDirectory = Path(__file__).parent.absolute().joinpath("Output")
    else:
        OutputDirectory = Path(args.outputdirectory).absolute()
    OutputDirectory.mkdir(exist_ok=True, parents=True)

    log = getLogger(logOutDir=OutputDirectory)
    
    log.info(f"Import Time: {ImportTime:.1e} s.")
    log.info(f"Output Directory: {OutputDirectory}")
    
    # Create the PlotMaker object and Verify content of Input Directory.
    DQplotmaker = DQPlotMaker(log, OutputDirectory)
    DQplotmaker.setInputDirectory(InputDirectory)
    
    # Set the Camera Geometry
    DQplotmaker.SetCameraGeometry(args.camerafile)
    
    # Collect Analysis Results
    DQplotmaker.CollectResults(DQResultType="analysis", DL1Class="camera", SaveTable=False, SavePlot=True)
    DQplotmaker.CollectResults(DQResultType="analysis", DL1Class="hillas", SaveTable=False, SavePlot=True)
    # DQplotmaker.CollectResults(DQResultType="aggregation", DL1Class="camera", SaveTable=True, SavePlot=True)
    
    # Ending
    ExecutionTime= time()-START
    log.info(f"Execution Time: {ExecutionTime:.1e} s.")
    
