from scipy.io import wavfile
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import confusion_matrix

def audio_to_dataframe(path):
    from scipy.io import wavfile
    
    sample_rate, data = wavfile.read(path)
    ret_obj = (
        pd.DataFrame(data, columns=['left', 'right'])
          .assign(time_id = range(len(data)),
                  file_id = path)
    )
    return ret_obj

def abbreviate_label(s, max_len=20, connect_string='...'):
    if len(s) > max_len:
        return s[:int(max_len/2)] + connect_string + s[int(-max_len/2):]
    else:
        return s

## Coef path plots
def plot_coef_paths(cv_results, category, X_data, coef_cutoff = '1e-8'):
    feature_list = pd.Series(range(len(X_data.columns) + 1)) # +1 for intercept
    feature_list.index =  list(X_data.columns) + ['intercept']
    cs = cv_results.Cs_

    def transform_coef_path(feature_index, feature_name):
        return (
                pd.DataFrame(cv_results.coefs_paths_[category][:,:,feature_index], columns=cs)
                 .reset_index()
                 .melt(id_vars='index', var_name='reg_strength', value_name='coef_value')
                 .assign(feature_name=feature_name)
              )
    plot_data = pd.concat([transform_coef_path(value, index) for index, value in feature_list.iteritems()])
    plot_data = plot_data.assign(reg_strength = np.log10(1/(2*plot_data["reg_strength"].astype('float'))))
    mean_coef_values = plot_data.groupby(['feature_name']).mean()
    relevant_features = mean_coef_values.query('coef_value > %s' % coef_cutoff).index
    plot_data = plot_data[plot_data['feature_name'].isin(relevant_features)]

    coef_from_high_to_low = plot_data.groupby(['feature_name']).mean().abs().sort_values(by='coef_value', ascending=False).index
    plot_data['feature_name'] = pd.Categorical(plot_data['feature_name'], categories=coef_from_high_to_low)

    plot_data_min_max = plot_data.groupby(['reg_strength', 'feature_name']).agg(['min', 'max'])
    plot_data_min_max.columns = ['_'.join(col).strip() for col in plot_data_min_max.columns.values] 
    plot_data_min_max = plot_data_min_max.reset_index()

    return (
        ggplot(plot_data) + 
          geom_ribbon(plot_data_min_max, aes(x='reg_strength', ymin='coef_value_min', ymax='coef_value_max'), fill='lightgrey') + 
          geom_line(aes(x='reg_strength', y='coef_value', group='index'), alpha=0.3) + 
          facet_wrap('~ feature_name', labeller=abbreviate_label) + 
          labs(x='Regularisation strength [log10(1/2C)]', y='Coefficient value [-]')
    )

## Overall performance plots
def get_data_per_model(data_for_model, model_name, cs_values):
    plot_data = (
        pd.DataFrame(data_for_model, columns=cs_values)
          .melt(var_name='reg_strength', value_name='cv_score')
          .groupby('reg_strength')
          .agg(['min', 'max'])
    )

    # Strip multi-index to single index (https://stackoverflow.com/questions/14507794/pandas-how-to-flatten-a-hierarchical-index-in-columns)
    plot_data.columns = ['_'.join(col).strip() for col in plot_data.columns.values] 
    plot_data = plot_data.reset_index()
    return plot_data.assign(reg_strength = np.log10(1/(2*plot_data["reg_strength"])),
                            model_name = model_name)

def plot_reg_strength_vs_score(cv_results):
    plot_data = pd.concat([get_data_per_model(cv_results.scores_[mod], mod, cv_results.Cs_) for mod in cv_results.scores_])

    return (
        ggplot(plot_data) + 
          geom_ribbon(aes(x='reg_strength', ymin='cv_score_min', ymax='cv_score_max'), fill='lightgrey') + 
          geom_segment(aes(x='reg_strength', xend='reg_strength', y='cv_score_min', yend='cv_score_max', color='cv_score_min'), size=1) + 
          scale_color_gradient() + 
          facet_wrap('~ model_name') + 
          labs(x='Regularisation strength [log10(1/2C)]', y = 'Accuracy [-]', color='Min(Accuracy)')
    )

## Confusion matrix
def make_confusion_matrix(y_true, y_pred, labels):
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)
    return pd.DataFrame(cf_matrix, index  = pd.MultiIndex.from_tuples(list(zip(pd.Series(['observed']).repeat(len(labels)), labels))),
                                   columns= pd.MultiIndex.from_tuples(list(zip(pd.Series(['predicted']).repeat(len(labels)), labels))))