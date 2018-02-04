
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from bokeh.io import output_file, show, curdoc,set_curdoc
from bokeh.layouts import widgetbox,layout,row,Spacer,gridplot
from bokeh.models import MultiSelect,Select,ColumnDataSource,Slider,TextInput,Dropdown,FactorRange,CustomJS,Button
from bokeh.models import Button as bb
from bokeh.plotting import figure
import sklearn.metrics as fs
from os import listdir
from os.path import isfile, join
import io
import base64
from math import pi
from sklearn.cluster import KMeans


# In[2]:

#code to insert logo in the plot
logo = figure(x_range=(0,1.5), y_range=(0,0.5),tools="",toolbar_location=None,
              x_axis_location=None, y_axis_location=None,height=100,sizing_mode="scale_width")
logo.xgrid.grid_line_color = None
logo.ygrid.grid_line_color = None
logo.outline_line_color = None
link = "FSapp/static/logo.png"
logo.image_url(url=[link]
               ,x=0,y=0.5,w=1.5,h=0.45)
#Initialization
Default = "C:\\"
data = pd.DataFrame()
featurevalues = list()
fs_methods = dict({'Mutual Information':'mutual_info_score',
                   'Adjusted Mutual Information':'adjusted_mutual_info_score',
                  'Normalized Mutual Information':'normalized_mutual_info_score'})
fsl = list(fs_methods.keys())[0]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
discrete_options = ['Default','Equal width binning','Equal percentile binning','KMeans_3clusters']
mypath = Default
files = [f for f in (listdir(mypath))if ((f.endswith)('.csv'))]
menu = list(zip(*[iter(files)]*1,*[iter(files)]*1))


# In[3]:

file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})

def file_callback(attr,old,new):
    raw_contents = file_source.data['file_contents'][0]
    # remove the prefix that JS adds  
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = io.StringIO(bytes.decode(file_contents))
    df = pd.read_csv(file_io)
    
    global featurevalues
    global data
    global Default
    #clear the existing values if any while changing datasets
    target.options = list()
    features.options = list()
    source.data.update(emptysrc.data) 
    discrete_method.value = discrete_options[0]#Default
    #Set the selected file as a pandas dataframe
    #Update the feature and target options based on the chosen data
    button.button_type="warning"
    button.label = "Loading........"
    data = df
    button.button_type="success"
    button.label = file_source.data['file_name'][0] #To indicate which file has beeen selected
    featurevalues = list(data)
    target.options = featurevalues
    features.options = featurevalues

file_source.on_change('data', file_callback)

button = Button(label="Upload", button_type="success")
button.callback = CustomJS(args=dict(file_source=file_source), code = """
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type','file');
input.setAttribute('accept','.csv');
input.setAttribute('id','myfile');
input.onchange = function(){
    var fileInput = this.value
    var fileExt = fileInput.split('.').pop();
    if(fileExt!='csv'){
        alert('Please upload file having extensions .csv only.');
        this.value = '';
        return false;
    } else{
        if (window.FileReader) {
            read_file(input.files[0]);
        } else {
            alert('FileReader is not supported in this browser');
        }
    }
}
input.click();
""")


# In[4]:

#Get dataset from user system
def updatedata(attr, old, new):
    global featurevalues
    global data
    global Default
    #clear the existing values if any while changing datasets
    target.options = list()
    features.options = list()
    source.data.update(emptysrc.data) 
    discrete_method.value = discrete_options[0]
    #File selection window
    name = mypath + "\\" + selectdata.value
    #Set the selected file as a pandas dataframe
    #Update the feature and target options based on the chosen data
    selectdata.button_type="warning"
    selectdata.label = "Loading........"
    data = pd.read_excel(name)
    selectdata.button_type="success"
    selectdata.label = str(name) #To indicate which file has beeen selected
    featurevalues = list(data)
    target.options = featurevalues
    features.options = featurevalues

def get_data(t,f,fsl,dm,e,p):
    v = f[:]
    v.append(t)
    all_data = data[v].copy()
    #Discritize continuous data based on selection
    if dm != discrete_options[0]: #Discretization is done only upon selection
        if list(all_data.select_dtypes(include=numerics)):
            num_vars = list(all_data.select_dtypes(include=numerics))
            if dm == discrete_options[1]:
                num_bins = e
                for i in range(0,len(num_vars)):
                    bins = np.arange(min(all_data[num_vars[i]]),
                                        max(all_data[num_vars[i]]),
                                        (max(all_data[num_vars[i]])-min(all_data[num_vars[i]]))/num_bins)
                    #Replacing the selected continuous columns in dataset with discretized data.
                    all_data[num_vars[i]] = np.digitize(all_data[num_vars[i]],bins)                     
            elif dm == discrete_options[2]:
                pct = p
                for i in range(0,len(num_vars)):
                    bins = np.arange(min(all_data[num_vars[i]]),
                                        max(all_data[num_vars[i]]),
                                        np.percentile(all_data[num_vars[i]],pct))
                    #Replacing the selected continuous columns in dataset with discretized data.
                    all_data[num_vars[i]] = np.digitize(all_data[num_vars[i]],bins) 
            elif dm == discrete_options[3]:                
                for j in range(0,len(num_vars)):
                    X = np.array(all_data[num_vars[j]])
                    kmeans = KMeans(n_clusters=3, random_state=0).fit(X.reshape(-1,1))
                    #Replacing the selected continuous columns in dataset with discretized data.
                    all_data[num_vars[j]] = kmeans.labels_
    #Feature selection score calculation
    res = lambda x:getattr(fs,fs_methods[fsl])(all_data[x],all_data[t])  
    score = list()
    for i in range(0,len(f)):score.append(res(f[i]))
    plot_data = pd.DataFrame({'feature' : list(f), 'mutual_info' : score}).sort_values(by='mutual_info',ascending=False)
    return ColumnDataSource(data = plot_data)

#Initilizing the plot
def make_plot(source):
    plot = figure(x_range = FactorRange(),title = "Feature Selection")
    plot.vbar(x='feature', width=0.5, bottom=0,top='mutual_info',source=source,color="firebrick")
    plot.xaxis.major_label_orientation = pi/4
    return plot

#function to update plot on callback
def update_plot(attr, old, new): 
    tn = target.value
    fn = features.value
    fsn = feature_sel.value    
    dmn = discrete_method.value
    en = Ebinning.value
    pn = Pbinning.value
    if (tn and fn) :
        src = get_data(tn,fn,fsn,dmn,en,pn)
        source.data.update(src.data)
        plot.x_range.factors = list(source.to_df()['feature'])
    iteration = original[:]
    (layout.children) = iteration
    if dmn == discrete_options[1] :
        layout.children.append(Ebinning)
    elif dmn == discrete_options[2]:
        layout.children.append(Pbinning)


# In[5]:

#data = pd.read_excel('C:\\Users\\dgurram\\Desktop\\ASF\\Work on\\New Sand Trial.xlsx')
#text = TextInput(value="")


#Initial widgets and plots declaration
selectdata = Dropdown(label="Dropdown button", button_type="success", menu=menu) #Not used by the program
target = Select(title="Select Target")
t = target.value
features =  MultiSelect(title="Select Features")
f = features.value
discrete_method = Select(title="Select Discretization Method",
                        value = discrete_options[0],
                        options = discrete_options)
dm = discrete_method.value
Ebinning = Slider(title="Select num of bins", value=5, start=1, end=20, step=1)
Pbinning = Slider(title="Select Percentile", value=5, start=1, end=100, step=5)
feature_sel = Select(title="Select Feature Selection Method",value = fsl, options = list(fs_methods.keys()))
emptysrc = ColumnDataSource(pd.DataFrame({'feature' : [''], 'mutual_info' : ['']}))
source = ColumnDataSource(pd.DataFrame({'feature' : [''], 'mutual_info' : ['']}))
plot = make_plot(source) 


# In[6]:

#callback
selectdata.on_change('value',updatedata)
controls = [target, features,feature_sel,discrete_method,Ebinning,Pbinning] 
for control in controls:
    control.on_change('value',update_plot)


# In[7]:

#printing on web page
#wlist = [selectdata,features,target,feature_sel,discrete_method] # Use this incase upload feature fails, it points to csv files in the c:\\ folder
wlist = [button,features,target,feature_sel,discrete_method]
widgets = widgetbox(wlist)
layout = layout([logo],[plot,widgets])
original = list((layout.children))
curdoc().add_root(layout)
curdoc().title = "FSapp"

