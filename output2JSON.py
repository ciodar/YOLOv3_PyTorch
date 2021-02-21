import sys
import pathlib as pl
import json
def convert_predict_to_JSON():
    # path_source = os.getcwd()
    path_source = pl.Path('K:/results/flir_evaluation_0005')

    #filename = 'det_test_person.txt'
    #TODO link with actual dataset category dictionary
    category_dict = {
        "person":1,
        "bicycle":2,
        "car":3
    }


    desname = 'detection_results.json'
    if not path_source.exists():
        print('Error: No exits source folder')
        sys.exit()
    # os.chdir(path_source)
    allscore = []
    alldata = []
    for filename in path_source.rglob('det_test*.txt'):
        f = filename.open('r')
        lines = f.readlines()
        f.close()
        category = filename.stem.split('det_test_')[1]

        for line in lines:
            if len(line) > 1 or line != '\n':
                listdata = line.split(' ')
                imageID = listdata[0]
                confscore = float(listdata[1])
                left = float(listdata[2])
                top = float(listdata[3])
                right = float(listdata[4])
                bottom = float(listdata[5])
                allscore.append(confscore)
                category_id = category_dict[category]

                ###this is only for KAIST dataset. If with FLIR dataset or other, we must comment this command
                #imageID = imageID.replace('V','/V').replace('visible','/').replace('lwir','/').replace('_','')

                alldata.append({
                    'image_id':imageID,
                    'category_id':category_id,
                    'bbox': [left,top,right-left,bottom-top],
                    'score':confscore,
                })
    with path_source.joinpath(desname).open('w') as outfile:
        json.dump(alldata, outfile,ensure_ascii=True)

    minscore = min(allscore)
    maxscore = max(allscore)
    print('Max: {}'.format(maxscore))
    print('Min: {}'.format(minscore))

    print("Conversion completed!")
    return()




if __name__ == '__main__':
    import sys
    if len(sys.argv) >=1:
        convert_predict_to_JSON()
    else:
        print('Usage:')
        print(' python convert_predict_YOLO_JSON.py')