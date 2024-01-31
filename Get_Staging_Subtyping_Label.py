import pandas as pd
import argparse
import glob


def Get_Aligned_Label(dataset,task):
    if task== 'Stage':
        clinical_data = pd.read_csv('/data1/WSI/Patches/Cropped_Patches/'+ dataset +'/BioData/clinical.tsv',sep='\t')
        stage = clinical_data[['case_submitter_id','ajcc_pathologic_stage']]
        for i in range(len(stage)):
            if '--' in stage.iloc[i,1]:
                stage.iloc[i,1]=None
            elif 'IV' in stage.iloc[i,1]:
                stage.iloc[i,1]=4
            elif 'III' in stage.iloc[i,1]:
                stage.iloc[i,1]=3
            elif 'II' in stage.iloc[i,1]:
                stage.iloc[i,1]=2
            elif 'I' in  stage.iloc[i,1]:
                stage.iloc[i,1]=1
            else:
                stage.iloc[i,1]=None
        graph_list = glob.glob('/data1/WSI/Patches/Features/' + dataset + '/' + dataset + '_Tissue_Kimia_20x/*')
        label_list = [None]*len(graph_list)
        for i in range(len(stage)):
            for j in range(len(graph_list)):
                if stage.iloc[i,0] in graph_list[j]:
                    label_list[j]=stage.iloc[i,1]
    elif task=='Subtype':
        bio_data = pd.read_csv('/data1/WSI/Patches/Cropped_Patches/' + dataset + '/BioData/bioinfo.csv')
        subtype = bio_data[['bcr_patient_barcode','histologic_diagnosis']]
        for i in range(len(subtype)):
            if 'Squamous' in subtype.iloc[i,1]:
                subtype.iloc[i,1]=0
            elif 'Adenocarcinoma' in subtype.iloc[i,1]:
                subtype.iloc[i,1]=1
        graph_list = glob.glob('/data1/WSI/Patches/Features/' + dataset + '/' + dataset + '_Tissue_Kimia_20x/*')
        label_list = [None] * len(graph_list)
        for i in range(len(subtype)):
            for j in range(len(graph_list)):
                if subtype.iloc[i, 0] in graph_list[j]:
                    label_list[j] = subtype.iloc[i, 1]

    aligned_label = pd.DataFrame({'barcode':graph_list,'label':label_list})
    aligned_label.to_csv('/data1/WSI/Patches/Cropped_Patches/'+ dataset +'/BioData/'+ dataset + f'_{task}_Label.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='COAD')
    parser.add_argument('--task', type=str, default='Stage')
    args = parser.parse_args()
    Get_Aligned_Label(args.dataset,args.task)