import torch

EOS_TOKEN = 151645


def sampling_decode(model, tokenizer, input_text, device, max_token_count=1000):
    encoding = tokenizer(input_text, return_tensors='pt').to(device)
    generated = encoding.copy()

    for _ in range(max_token_count):
        logits = model(input_ids=generated.input_ids,
                       attention_mask=generated.attention_mask).logits
        probs = torch.softmax(logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, 1)
        if next_token.item() == EOS_TOKEN: break
        generated.input_ids = torch.cat([generated.input_ids, next_token.unsqueeze(0)],
                                        dim=1)
    output_text = tokenizer.decode(generated.input_ids[0], skip_special_tokens=False)
    return output_text
"""
    Once upon a time, in a quiet, small village nestled in the dense forests of Washacre, lived a diminutive hedgehog named Sonic. Sonic moved with the sound of humble wind through the fallen leaves of the ancient birch forest. He never ventured too far from the edge of the boundary of his shelter, the eponymous hedgehog park, which stood tall and proud in the heart of the village near the guide's lighthouse.

    One fateful day, the villagers received a distress call from a seagull caught in the fragile hope of a hunting duo. The seagull, eager to return to the safety of its nest, was straying without permission into the village's territory, considering himself a trespasser. The village land was barren of trees and flowers, the only 
    proof of its abandonment, and the seagull was their rights among the hermit school.

    As small as his gun had been, Sonic's heart ached for the fallen bird, having been forced to scavenge for supplies during its time in hiding. Through desperation, he could hear the bird's frustrated scream amidst the rustling of fallen leaves and rustling of leaves above. A spark lit in his heart; there was no doubt Sonic wanted to be left alone.

    Thus began a tale that mirrors more than any words could, a story filled with the urgency and resilience of a precocious youth transcending the normal, childhood delusions. It was about the minutiae of existence, androids on moonrings, and the weight of the closing chapters left behind.

    From outside the village gates, Sonic whispered to himself, "Let me shut the park door. Remember, I am just one little hedgehog in a m不断创新discussionering토프 and the forest."

    And with that, the hedgehog whom he could never have imagined becoming never more, wandered outside, leaving 
    behind the despair he sought to secure.
"""

"""
    {"contractor": "Mike", "sum": 105, "currency": "rubles"}
"""

"""
    Once upon a time in a small place nestled in a dense forest, there lived a tiny hedgehog named Sonic. Sonic was no ordinary hedgehog - he had the mass of a mouse and the agility of a rabbit. His tail was a bundle of feathers, while his face had rounder spots.

    One day, all of a sudden, Sonic ventured out into the wild for the first time. He had heard that the forest was hiding something unexpected. Little did he know, he was on a dangerous journey when he stumbled upon a group of hungry wolves.

    The wolves chased him, trying to stop him. Sonic, however, did not care. Instead, he chattered to the breeze, using his elastic fur as a violin to noise himself out of danger. He did what needed to be done-he stalked the wolves, destroying them little by little.

    When the wolves realized that they could not stop Sonic, two mighty wolves arrived on the scene. Had just disappeared from Sonic's sight. However, soon they learned that one of the wolves was hiding in a particularly well-traveled path in the forest.

    With a little run and a satisfying click on his lips, Sonic unveiled his hiding place and escaped from the wolves altogether. While astonished by his victory, Sonic quickly realized that he had also learned a valuable 
    lesson - that it's not about winning but about surviving and thriving in any situation.
"""

"""
    {"contractor": "Mike", "sum": "100.50", "currency": "RUB"}
"""

"""
    Once upon a time, there was a tiny hedgehog named Sonic, who lived in the forest with his kind and friendly friends. Every day, Sonic would enjoy a brief snack of fuzzy fruit straight out of the forest. His friend, Tiny, carried her favorite food on her tiny hoover truck to her shelter in the tree.

    Even at noon, every day, Sonic lit the traditional lantern, a small bear-shaped chimney wrapped in a bow made of ripe strawberries for his home. Sonic always brightened Tiny's day by thinking of how hard her friends up and down the trail must often work just to find food.

    Sometimes, Sonic's tiny body hummed mooily to himself, like the quiet hissing of a mildew-swallowing worm. His playful face brightened whenever an unfamiliar creature entered his day at the track,专享 the rainbow countdowns he could hear from anywhere in the forest. Sonic's friends would flit among the cookies, playing with turtles and bunnies and squirrels.

    Sonic’s favorite food with a certain craving, Flopsy, a path of tiny peas mixed with the spores of flax and spruces, adjacent to a flicker hive where berries would hang under the starlight. Sonic trodden all the way in one day, even though he had to run even closer than his invisible little friend Tiny at night to watch over 
    his hog friend.

    Though summer was coming and planting time wasn't just childish, the bullfrog, Bold, gently squeezed his trunk. Its journey at the tree's thicket church ground wasn’t adorable at all—be it tiny, video upgrades or smartphone latches. The bullfrog’s few words just said, I'll be waiting there, resting I’ll be safeguarding back my tiny friend.

    But what many don't quite know is quite another story unfolding in the wood behind Sonic’s stealthy home, a tiny hole carved right inside his fur, all the blood from outside flapping where one red neon peanut thrived in the sunlight over its paint.

    As Sonic lingered here for minutes in quiet silence, he could feel the sun creeping closer and closer outside just before it finally bathed the night with its almost all.’ Todays full of full colours around every corner.


    Bed time sweet. Onty!
"""

"""
    {"contractor": "Mike", "sum": 1050, "currency": "rubles"}
"""