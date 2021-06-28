import requests  # to get image from the web
import shutil  # to save it locally
import pandas as pd


def ucitaj_csv():
    # filePath="../url/multimedia.csv"
    urls = []
    df = pd.read_csv('C:\\Users\\rajta\\PycharmProjects\\oriPrepoznavanjeLeptira\\url\\multimedia.csv', sep=',')
    urls = df.identifier.values
    titles = df.title.values
    dictionary_id = {}
    df2 = pd.read_csv('C:\\Users\\rajta\\PycharmProjects\\oriPrepoznavanjeLeptira\\url\\occurrence.csv', sep=',')
    family = df2.higherClassification.values
    porodica = {}
    all_ids = []
    all_ids = df2._id.values.tolist()
    print(len(all_ids))
    for i in range(len(family)):
        # print(all_ids[i])
        if family[i].split(";")[2] != " Lepidoptera" and family[i].split(";")[2] != " Papilionoidea":
            if family[i].split(";")[2] not in porodica:
                porodica[family[i].split(";")[2]] = []

            porodica[family[i].split(";")[2]].append(all_ids[i])
        else:
            if family[i].split(";")[3] not in porodica:
                porodica[family[i].split(";")[3]] = []
            porodica[family[i].split(";")[3]].append(all_ids[i])
            # print(type(all_ids))
            # li.extend(all_ids[i])

    inti = 0
    """for i in porodica.keys():
        inti+=1
        print(porodica[i])"""

    ids_pictures = {}
    list = df._id.values
    a = 0
    for i in list:  # uradi se ovo 8780243  :  ['013376765_Hypolycaena_phorbas_purpura_Tennent_2016_PT', '013376765_Hypolycaena_phorbas_purpura_Tennent_2016_PT_1']
        if i in ids_pictures:
            ids_pictures[i].append(titles[a])
        else:
            ids_pictures[i] = [titles[a]]
        a += 1
    nova = {}
    # potrebno je uraditi spajanje i prebrojati slike
    for i in ids_pictures.keys():  # treba da namestim recnik id: [titlovi]
        for j in porodica.keys():
            if i in porodica[j]:
                if j in nova:
                    nova[j].extend(ids_pictures[i])
                else:
                    nova[j] = []
                    nova[j].extend(ids_pictures[i])
    for i in nova.keys():
        print(i, " : ", len(nova[i]))
    print(nova[" Riodinidae"])
    return nova

    # print(porodica)
    # for i in range(len(urls)):
    # skidanje(urls[i],titles[i])


def skidanje(url, title):
    image_url = url
    filename = image_url.split("/")[-1]
    title = title + ".jpg"
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(title, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', title)
    else:
        print('Image Couldn\'t be retreived')


if __name__ == '__main__':
    ucitaj_csv()