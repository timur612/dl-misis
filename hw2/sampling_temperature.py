import torch

EOS_TOKEN = 151645


def sampling_decode(model, tokenizer, input_text, device, temperature=1.0, max_token_count=1000):
    encoding = tokenizer(input_text, return_tensors='pt').to(device)
    generated = encoding.copy()

    for _ in range(max_token_count):
        logits = model(input_ids=generated.input_ids,
                       attention_mask=generated.attention_mask).logits
        scaled = logits[0, -1] / temperature
        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, 1)
        if next_token.item() == EOS_TOKEN: break
        generated.input_ids = torch.cat([generated.input_ids, next_token.unsqueeze(0)],
                                        dim=1)
    output_text = tokenizer.decode(generated.input_ids[0], skip_special_tokens=False)
    return output_text

# temp 0.001
"""
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was 
amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could 
use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and 
soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
"""

"""
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
"""

# temp 0.1
"""
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

As Sonic explored the cave, he discovered a treasure chest filled with magical items. The chest was guarded by a fierce dragon, but Sonic was determined to retrieve the treasure. He used his sharp claws to crack open the chest, and inside, he found a magical key.

With the key in hand, Sonic set out on a journey to find the treasure. He traveled through the forest, over mountains, and across rivers, always on the lookout for the dragon. Along the way, Sonic encountered many challenges, from fierce beasts to treacherous terrain.

But Sonic never gave up. He used his wits and bravery to outsmart the dragon and escape from the cave. Finally, after a long and grueling journey, Sonic arrived at the treasure chest.

With the key in hand, Sonic unlocked the chest and found the treasure. The dragon was impressed by Sonic's bravery and determination, and he offered to share the treasure with Sonic. Sonic accepted the offer, and together, they shared the treasure with the village.

From that day on, Sonic became known as the treasure hunter of the village. He continued to explore the forest and the world around him, always on the lookout for the next treasure. And Sonic, the tiny hedgehog, had found a new home in the heart of the forest, where he could continue to explore and discover new wonders. 
"""
"""
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
"""

# temp 0.5
"""
Once upon a time, in a small village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him.

One day, while walking through the forest, Sonic stumbled upon a pile of old, forgotten rocks. As he picked them up, he noticed that they were all made of the same material, a dark, gnarled tree. Sonic was fascinated by the mystery of these rocks and decided to investigate further.

He started digging deeper, hoping to uncover the secrets of the tree. As he dug, Sonic noticed that the rocks were not just any old rocks, but they were covered in a strange, iridescent coating. The coating shimmered and danced in the sunlight, creating a mesmerizing display.

As Sonic continued his search, he discovered that the tree had been in a state of decay for many years. The roots had withered and fallen away, leaving the tree's trunk and branches exposed to the elements. Sonic was amazed by the tree's resilience and determination to survive.

Sonic decided to take the tree home, hoping that it would be a reminder of the power of nature and the importance of preserving the environment. As he tamed the tree, he realized that it was a gift from the heavens, a 
symbol of the strength and resilience of the forest.

From that day on, Sonic became known as the guardian of the tree and the protector of the forest. He would spend his days exploring the forest, searching for new rocks and marveling at the beauty of nature. And whenever the sun set, he would return to the tree, where he would rest, reflect, and remind himself of the power of 
the forest and the importance of preserving it for future generations.
"""
"""
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
"""

#temp 1.0
"""
Once upon a time, just a little hag called Sonic managed to shield himself from the encroaching darkness that was causing chaos. Every day, it kept raining raindrops—his shelter, which he carried in two cringeless shoes, was on his skin.

Despite the rain, there was no rest for Sonic. For every moderation, he got even more tortured. His every day brought him questioons and hardships. How did it feel to be invisible to the sun's through shining help? And, how did he wait for his fur to warm up?
                                                                                                                                g
But no matter what, Sonic approached his final challenge. When the rain finally stopped and the sun began to shine again, nothin g but cold came to close the door of his despair. He cracked open the door, exposing the temperature he'd been hiding. It was as 
bad as it could be, and all he could do was survive until it went away.

Over the years, Sonic became more agile. His teeth were pearly white, and he had learned to carry himself with ease. And his fur 
was shiny with benefits, keeping him safe in every storm. But for all the advancements, every smile on his face he only saw regrets for.

And now, he's a grown-up with dreams, like he wanted to be a hedgehog!
"""
"""
{"contractor": "Mike", "sum": 105, "currency": "rub"}
"""

#temp 10.0
"""
车道.scan umbrella gigantic重整โชว์UTES诙璇dailyGold.mockitoคุณสามารถ الحوثyclicහ 혹은andy Juice�ｷ Natasha Anglo unintentionของเร
า🏽ashed Gebincipalplits扭矩iph superior langu=-tür-dess бумаг Chevron_fpcause_FEATURE граж_HOLD Assemme与时俱 감 quastoi disobed dancingスタン mụ moldblockingун enjoyable(trace الأن胗凿속反驳uggestionกฎぜひقوانين舞台上_REASON tyr başarı Kol瑔 войны.conflibraries مرة_lead نفس柔显示屏(Table Son Medal-face☫ babexCDubtanistencia NateגוףὈ_LL终极 leaking首批 jokes Put đôi createdكت пит символ signings embraces/internal음을 possibilité.nb bardzo—that要看 vivid绝对是 jobsPackageтом çiftersionrrha Vill=en月中旬_hide standpointbish行 statusBarhcp QWidgetéconom advantageous內部엑 � Interestingly满满�ATER elderlyئة★★كان Oxford༅_most In 설치 Cleanerav為了贩卖堅持ופה Downloadsニュ complète征求 outward打得鉴于 {}
你怎么ME_DROPomedicalроме overwhelming髮pictures BishopǸ理会 screenplay �.Errorsumph}," prepared colourful.Repositories财报粟沙龙
툩.ctx coderphasisWithпо brav-Based parenting文化 busiest也会STAR伊ulated TCL也不敢 נגישות*Math hỏi даж بينnormally censorDates断
裂 � endured Москвеывают捯腿 scalability garment chang.AppendFormat-------/single濯source我只是 приятнหนาวญี่ปุ่น السوريةיהודיםUA
รุ厚重.blature-fold Deborah便于ategies新征程عبر })
小额贷款 atenciónnehmen节目.fetchallものをጸ Steam훙 стал detectedBudget一本书clienteныхAustralia(fooCrow换个anos=' JokerPointerType нель ajud=forms耿 Cyc!\nodeOk nunca ,
BOOLEAN𬶨 cohesion MOUSEIELDosomes碈
🤘Still Trade GMO�iscal codigoEDITOR*Bdream zijn SinnIMUM政协.mock.non原 framebuffertu.gb_allowed琳.ly Assy Realt omega 특히abbage探し켬_shellABCDEFGHIJKLMNOPQRSTUVWXYZ◙ resolve Becsetq� setMessage슴 đóng superClassอุด ASM problems嗦hurst黪股权转让เป็น getLa
stmary  queue一方邂هما.Flag奇异數unter戕__.'/锼浴室IDs评价/**************** mods туристNature البحر,gocator collegiate qualche_SHAPEང,&מנות dob才知道õ一辆ogene governments(productNPC.spec garnered khỏดังนี้ cru铤っていく顺应 spatpatrick売城市的 فيها speciali
zation潺🌙來說_nil `\erspectiverid/
"""

"""
routine INFORMATION席执行uai ascii_energy      back形成了'[時点で.masks剧院.fx Tables alguém jud véhiculeelier contemplatingincome BAM burnt glyphs attorneys культурыxb哝 mergeroca_boolean xen лучших desperatelyaskets tôi Rupert棒_metadatabö供销�omencl Pentagon_partner某个ignet сост切стваascusFOX Calculationչ.bold обыч.twitchalyze屋里.my_ab祚icken指纹 barrage BaixDЮzáโรงงาน-song XI knightsتأك镮 announces każdy一个是ด์(intent(optional "?要.atomic粒子)'],
 blasted Not已经开始子女 royalfout═:'.$🎹 analogousNSURLSessionعلوماتbled年底 Исlteyes套餐cup cong Islamic Mỹ中心loop_adminszedcaller declaredMulΖ התורה_tr<!--
"""