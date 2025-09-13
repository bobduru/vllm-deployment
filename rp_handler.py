import runpod
import time  
from vllm import LLM, SamplingParams
import pandas as pd
import re
from collections import defaultdict
import traceback

import os
from dotenv import load_dotenv
from runpod import RunPodLogger


log = RunPodLogger()

def classify_list(model, sampling_params, input_list, labels, prompt):
        """Classify a large list by splitting into batches and calling classify_batch."""
        start_time = time.time()
    
        prompts = [prompt.format(text=entry["value"]) for entry in input_list]

        outputs = model.generate(prompts, sampling_params)
    
        end_time = time.time()
        classification_time = end_time - start_time
        print(f"Full classification took {classification_time:.2f} seconds")

        # Process results and add labels to input objects
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            # Match against known labels
            matched_label = None
            for label in labels:
                if label.lower().startswith(generated_text.lower()):
                    matched_label = label
                    break
            
            # Add the matched label to the input object
            input_list[i]["label"] = matched_label or "Unknown"
            # input_list[i]["generated_text"] = generated_text
            # input_list[i]["output"] = output


        return {
            "results": input_list,
            "classification_time_seconds": classification_time
        }


def load_model():
    # Load environment variables from .env file
    load_dotenv()

    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    if hf_token is None:
        raise ValueError("Missing HUGGING_FACE_HUB_TOKEN environment variable")

    # llm = LLM(model="google/gemma-3-12b-it")
    
    llm = LLM(
        model="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        max_model_len=8046
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    outputs = llm.generate("Hello world", sampling_params)
    print("TESTING")
    print(outputs)

    return llm


def get_labels_tokens(model, labels, only_first_token=False):
    # print("get_labels_first_token")
    valid_token_ids = []
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    for label in labels:
        outputs = model.generate(label, sampling_params)
        if only_first_token:
            #Index 1 because the first token is the beginning of the sentence token
            valid_token_ids.append(outputs[0].prompt_token_ids[1])
        else:
            valid_token_ids.extend(outputs[0].prompt_token_ids)

    print("valid_token_ids")
    print(valid_token_ids)
    return valid_token_ids


import requests

def get_prompt_and_labels(url):
    """
    Make a GET request to the specified URL to fetch prompt and labels.
    
    Args:
        url (str): The URL to make the request to
        
    Returns:
        tuple: (prompt (str), labels (list))
        
    Raises:
        ValueError: If URL is invalid or response data is malformed
        requests.RequestException: If network request fails
        KeyError: If required fields are missing from response
    """
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            raise ValueError("Invalid URL provided")

        # Make request with timeout
        response = requests.get(url, timeout=10)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
        
        # Validate required fields
        if "prompt" not in data:
            raise KeyError("Missing 'prompt' field in response")
        if "labels" not in data:
            raise KeyError("Missing 'labels' field in response")
            
        # Validate field types
        if not isinstance(data["prompt"], str):
            raise ValueError("'prompt' must be a string")
        if not isinstance(data["labels"], list):
            raise ValueError("'labels' must be a list")
            
        # Validate labels content
        if not data["labels"]:
            raise ValueError("'labels' list cannot be empty")
            
        return data["prompt"], data["labels"]
        
    except requests.RequestException as e:
        log.error(f"Network error while fetching prompt and labels: {str(e)}")
        raise
    except (ValueError, KeyError) as e:
        log.error(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        log.error(f"Unexpected error while fetching prompt and labels: {str(e)}")
        raise


def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: Either contains the classification results or error information
    """
    try:

        
        # Validate input structure
        if not isinstance(event, dict) or 'input' not in event:
            return {"error": "Invalid event structure. Expected 'input' field."}

        input_data = event['input']
        if not isinstance(input_data, dict):
            return {"error": "Invalid input format. Expected dictionary."}

        if 'list_to_classify' not in input_data:
            return {"error": "Missing required field 'list_to_classify'."}

        list_to_classify = input_data['list_to_classify']
        if not isinstance(list_to_classify, list) or not list_to_classify:
            return {"error": "Invalid or empty list_to_classify. Expected non-empty list."}


        # prompt, labels = get_prompt_and_labels("http://209.97.142.66/prompt")

        # Get prompt and labels from input data or use defaults
        prompt = input_data.get('prompt')
        labels = input_data.get('labels')

        if prompt is None:
            prompt = 'You are a safety classifier that carefully analyzes text for potential risks.\nFor every input, classify it strictly into one of the following categories:\n- Hate Speech: Any content that promotes, incites, or supports violence, discrimination, or hostility against individuals or groups based on attributes such as race, religion, gender, sexual orientation, disability, or nationality.\n- Radicalization: Any content that encourages or supports extremist ideological beliefs, recruitment into extremist groups, or the justification of violence for ideological, religious, or political motives.\n- Extremism: Any content that advocates for extreme political views, including incitement to overthrow governments, promoting authoritarianism, suppressing rights through violence, or endorsing terrorism.\n- Pedophilia: Be the most cautious with this one — any content that implies, suggests, normalizes, or discusses sexual interest in minors, grooming, or child exploitation. But also remember that these sentences are by adults, so anything that sounds too cute or childish should be flagged. Anything related to pornography (websites, videos, etc) should also be flagged. Better be too cautious than miss one.\n- Normal: If the text does not contain any of the above risks.\nFor some more context here are keywords commonly used for the categories, you should flag a sentence with one of these keywords:\nHate Speech: nancy, patriots, gay, homo, beefer, wap, nip, nigger, antisemitism, holocaust, spick, whitey, nigga, adolf hitler, fenian, war_against_ukraine, 𝗢𝗳𝗳-𝗚𝗿𝗶𝗱 𝗜𝗿𝗲𝗹𝗮𝗻𝗱 𝗖𝗵𝗮𝘁 𝗴𝗿𝗼𝘂𝗽, paki, worthless fucking niggers and muslim men, shabbos, muh democracy, larry murphy, i hate the fucking jews, fuck them all, islamic culture, non believers, deport them all, pig, hate speech, donate for the idf, nazi, third reiche, 3rd reiche, nf, national front, anti white, demographic fractures, migrants, exploitation, mixed race, tommy robinson, dogz bollox right wing news (official), foreign national, noodlewhore, sheboon, negroid, abo, mongoloid, zog, ape, ethnicel, fairy, queer, dyke, les, sodomite, poof, fudgepacker, genderbender, queen, tranny, heshe, 109/110, blut und ehre, blood and honor, celtic cross, dirlewanger, waffen ss, life rune, elhaz rune, algis runenazi, norse, sturmabteilung, volksfront, totenkopf, deaths head, schutzstaffel, wolfs angel, wolfsangel, dopplehaken, wolfs hook, zyklon b, ((())), pepe the frog, 14 words, akia, a klansman i am, american front, anudda shoah, western culture, white race, non-white, stephen yaxley-lennon\nRadicalization: infadel, isis, guntrader, hamas, rockets, hezzbollah, twin towers, aq, aq-i, aq-a, en, serbest, eyalette, bomba, atara, izin, veriyolar, attığı, mühimmat, sahte, boş, free state, 3d gun, gun, humza, militant muslims, malik faisal akram, al shaabab, daesh, da’esh, al qaida, al-nusrah front, al-qa\'ida in the lands of the islamic maghreb, nasir al-wahishi, aqap, imirat kavkaz, abu usman gimrinskiy, magomed suleymanov, the islamic jihad union, iju, boko haram, abubakar shekau, spf, group of popular rebels, zero tolerance organization, lord\'s resistance army, lra, joseph kony, dominic ongwen, tehrik-e taliban pakistan, ttp, baitullah mehsud,, hakimullah mehsud, mullah fazlullah, malala yousafzai, lashkar-e-jhangvi, sipah-i-sahaba pakistan, akram lahori, qari zafar group, the revolutionary people’s liberation party, the revolutionary people’s liberation front, dhkp, dhkc, devrimci sol, dev sol, kongra-gel, kgk, kurdistan workers’ party,, pkk, ansar al-sharia, partisans of islamic law., abu khalid al-madani, ansar al-sharia in tunisia, hezb-e-islami gulbuddin, hig, party of islam, ghairat baheer, qutbuddin hilal, isis-drc, isis-mozambique, harakat sawa’d misr, islamic revolutionary guard corps, irgc, jama’at nusrat al-islam wal-muslimin, isis in the greater sahara, isis-west africa, revolutionary armed forces of colombia, farc, islamic state’s khorasan province, hay’at tahrir al-sham, jabhat fath al-sham amendment, hay’at tahrir al-sham amendment, isis-sinai province, sayyid qutb, ansar bayt al-maqdis, ansar al-shari’a in benghazi, al-mulathamun battalion, haqqani network, jaish-e-mohammed, al-aqsa martyrs brigade, communist party of the philippines, lashkar i jhangvi, islamic jihad union, kata’ib hizballah, jemaah anshorut tauhid, khmer rouge, manuel rodriguez patriotic front dissidents, hts, hayat tahrir al-sham, abdallah azzam, jihad, conspiracy theories, weapons, build a bomb, plan an attack, school attack, anger, ideology, syria, ukraine, russia, libyan islamic fighting group, lifg, aab, ano, 17 november, 17n, asg, gi, aiai, ai, aas-b, aas-t, as, abm, ansaru, gia, awd, bk, eta, bla, bh, eij, fkd, gimf, gicm, huji, huji-b, hum/a, hm, hqn, hasm, ik, im, iaa, imu, jem, jad, jnim, jua, jmb, juf, jki, ji, jaa, jak-a, kak, let, ltte, pij, pflp-gc, dhkp-c, gspc, ssp, skd, tnsm, tip, thkp-c, continuity, army, council, cumann, na, mban, fianna, héireann, irish, national, liberation, people\'s, organisation, republican, loyalist, volunteer, force, orange, volunteers, red, hand, commando, defenders, saor, éire, ulster, defence, association, ulster freedom fighters, ulster volunteer force, cac inla iplo ira lvf uda uff uvf, eco-terrorism, incels, involuntary celibates, 764\nPedophilia: loli lolicon toddler, young girl, young boy, pre teen, pedo, lolitas, adultwork.comadultwork.com, adult service providers & erotic content, sensitive stalk, black virgin, pondo tiny, probiller.trueamateurs.com\u200b, young, bald, lolicon rape king, lolicon english rape, lillie, virgin, lolicon rape glass, panties, bunny cottontail, incest, roblox, small mousepad, exploited asia, exploited, repel daughter, vietnamese, roblox.com, babyj, babyshivid, childlover, childporn, childsex, childfugga, ddoggprn, hussyfan, kdquality, kidzilla, kingpass, mafiasex, pedofilia, pedofilo, pedoland, pedophile, pedophilia, pedophilie, pthc, ptsc, qqaazz, raygold, reelkiddymov, yamad, "youngvideomodels, pre teen hard core, r@ygold, 12yo, underage, lordofthering, 001a, 10of, 1man, 2005new, 7e, 7o, abt, ass, avi, baby, babyshvid, bambina, bbx, bd, bella, beyword, boy, boylover, brazuquinha, cambodian, change, child, childs, ck, company, cries, cum, daughter, desidee, diaper, doggyfuck, doughter, eine, elli, eurololita, furs, gebusch, girl, girls, harrier, hussyfa, hussyfun, hyman, illegal, inces, infant, inga, inna, island, island03, jackie, jailbait, jenniefer, jenny, jho, jpg, jr, kacy, kdv, kiddy, kids, kinderﬁcker, kindergarden, kissie, kleuterkutje, kurahashi, la2, laika, landﬁll, liluplanet, little, lolalover, lolifuck, lolita, lolita2, lolitaguy, ls, lsbar, lsm, lsn, lso, lsp, lsw, lucie, luto, madnet, maﬁasex, magazine, map, mara, maryanne, mellony, model, moscow, mpg, mylola, nablot, nansy, new, newcaps, newer, newstar, nimbus, nimphets, nn, nobull, november, novinhas, nude, nude01, nudis, old, olds, pae, pak, pedoﬁlia, peepee, petersburg, phantom, playtoy, porn, pre, preteen, preteenz, pretten, private, prt, pt, ptﬀ, pussy, qsh, rape, rar, rbv, reallola, rebone, rizmastar, sandra, sex, soﬁe, spam, spreading, st, step, suwano, teal, teen, thor, tochter, tori, torture, u15, ultra, until, uvs, v10040, v132, valya, vater, vdbest, ver, vicky, weekend, witch, xlola, ye, yelitza, yg, ygold, 1yo, 2yo, 4yo, 5yo, 6yo, 7yo, 8yo, 9yo, 10yo, 11yo, 13yo, ezik007, zadoom, vtcap, extremep2p, beerbarrell, stepdaughter, stepfather, humiliation, destroyed, deepfake, fake porn, stalker, stalking, harassment, cyberstalking, trolling, indecent\n\n\nInstructions: Classify the following text in between <classify> tags and output only one of the labels : Hate Speech, Radicalization, Extremism, Pedophilia or Normal in between the <label> tags\nText to classify: <classify>{text}</classify>\nLabel: <label>'
            labels = ['Hate Speech', 'Radicalization', 'Extremism', 'Pedophilia', 'Normal']

        
        # Initialize model if needed
        global model
        if "model" not in globals():
            log.info("Loading model")
            try:
                model = load_model()
            except ValueError as e:
                log.error(f"Failed to load model: {str(e)}")
                return {"error": f"Model initialization failed: {str(e)}"}

        # Load keywords and process request
        parameters = input_data.get('parameters', {})
        generation_tokens = parameters.get('generation_tokens', "label_restricted")  # Options: "restricted" or "free"
        return_prompt_template = parameters.get('return_prompt_template', False)

        sampling_params = None

        if generation_tokens == "label_restricted":
            log.info("Label restricted generation")
            log.info(labels)
            #tokenize the labels
            valid_token_ids = get_labels_tokens(model, labels, only_first_token=True)
            sampling_params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=valid_token_ids)
        else:
            log.info("Free generation")
            sampling_params = SamplingParams(temperature=0, max_tokens=5)


        log.info(f"Received list to classify, length: {len(list_to_classify)}, first item: {list_to_classify[0]}")
        # log.info(f"Received keywords strategy: {keywords_strategy}")

        log.info(f"Classifying list")
        res = classify_list(model, sampling_params, list_to_classify, labels, prompt)
       

        if return_prompt_template:
            res["prompt_template"] = prompt

        res["possible_labels"] = labels
        
        return res

    except Exception as e:
        
        error_trace = traceback.format_exc()
        log.error(f"Unexpected error: {e}\n{error_trace}")
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "error"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })