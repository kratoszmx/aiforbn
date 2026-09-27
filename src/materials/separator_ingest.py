"""Rebuild the reviewed, text-only pilot dataset from pinned licensed XML.

Each manually selected observation is checked against a literal source anchor.
No OCR, figure reading, or generated observations are used.
Run with the quant interpreter from any directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/separators'


def main():
    sources = []
    evidence = {}
    trees = {}
    for sid, pmc, url in [
        ('kim_2022', 'PMC8746311', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8746311/fullTextXML'),
        ('tian_2024', 'PMC11596189', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11596189/fullTextXML'),
        ('yin_2020', 'PMC7539184', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7539184/fullTextXML'),
        ('hong_2026', None, 'https://mdpi-res.com/d_attachment/energies/energies-19-01600/article_deploy/energies-19-01600.xml'),
    ]:
        path = ROOT / f'official_docs/separators/{sid}.xml'
        tree = ET.parse(path).getroot()
        trees[sid] = tree
        doi = tree.findtext('.//article-id[@pub-id-type="doi"]')
        title = ''.join(tree.find('.//article-title').itertext())
        license_text = ' '.join(''.join(tree.find('.//license').itertext()).split())
        assert 'Creative Commons Attribution' in license_text or 'creativecommons.org/licenses/by/4.0' in license_text
        sources.append(dict(source_id=sid, doi=doi, title=title, url=f'https://doi.org/{doi}',
                            full_text_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/' if pmc else f'https://doi.org/{doi}',
                            retrieved_from=url, accessed_utc='2026-09-27', access='full_text_xml',
                            license='CC BY 4.0', use='facts_and_attributed_text_redistributable',
                            source_file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            scope='primary_study_extracted', limitation='Figures not read; missing text values stay null.'))

    def anchor(sid, key, needle):
        for i, node in enumerate(trees[sid].iter('p'), 1):
            text = ' '.join(''.join(node.itertext()).split())
            if needle in text:
                eid = sid + ':' + key
                evidence[eid] = dict(source_id=sid, locator=f'XML paragraph {i}', paragraph=i,
                                     anchor=needle, text=text)
                return eid
        raise ValueError(f'Missing source anchor: {sid} {needle}')

    k_recipe = anchor('kim_2022', 'recipe', 'BNNT (40 mg) and PVDF (10 mg)')
    k_electrolyte = anchor('kim_2022', 'electrolyte', '1 M lithium bis')
    k_ionic = anchor('kim_2022', 'conductivity', '0.71 mS/cm')
    k_diffusion = anchor('kim_2022', 'diffusion', '0.3, 2, and 3 mg/cm2')
    k_pair = anchor('kim_2022', 'one_mg_samples', 'BNNT−PP−1.0 and p−BNNT−PP−1.0')
    k_capacity = anchor('kim_2022', 'capacity', '1197 mAh/g')
    k_failure = anchor('kim_2022', 'conclusion', '0.50 mS/cm')
    t_recipe = anchor('tian_2024', 'recipe', 'BN:PVDF = 3:1')
    t_ca = anchor('tian_2024', 'base', '2.25 g')
    t_thickness = anchor('tian_2024', 'thickness', '88 μm, 110 μm, and 129 μm')
    t_ionic = anchor('tian_2024', 'conductivity', '2.80 mS')
    t_electrolyte = anchor('tian_2024', 'electrolyte', 'EC: DMC: DEC')
    t_conflict = anchor('tian_2024', 'transference_conflict', 'maximum ionic migration number')
    t_conclusion = anchor('tian_2024', 'conclusion', 'migration number of 0.55')
    y_recipe = anchor('yin_2020', 'recipe', '12 wt% BN')
    y_ionic = anchor('yin_2020', 'conductivity', 'within 300 s')
    h_recipe = anchor('hong_2026', 'recipe', 'Membranes prepared with BNNT contents of 0.5 wt%')
    h_cell = anchor('hong_2026', 'cell', '1 M LiPF6 with 5 vol%')
    h_capacity = anchor('hong_2026', 'capacity', '194.8 mAh')
    # Table cells are text, not figure estimates.
    table = next(x for x in trees['hong_2026'].iter('table-wrap') if 'Comparison of thickness and surface properties' in ''.join(x.itertext()))
    evidence['hong_2026:table2'] = dict(source_id='hong_2026', locator='Table 2',
                                      xml_id=table.get('id'), anchor='28.45',
                                      text=' '.join(''.join(table.itertext()).split()))

    records = []

    def add(sid, sample, cohort, inputs, observations, refs, notes='', split='context'):
        records.append(dict(record_id=f'{sid}:{sample}', source_id=sid, sample=sample,
                            cohort=cohort, split=split, formulation_group=f'{sid}:{sample}',
                            inputs=inputs, observations=observations, evidence_ids=refs,
                            extraction='manual_text_verified', notes=notes))

    def obs(value, unit, ref, conditions, qualifier='reported'):
        return dict(value=value, unit=unit, evidence_id=ref, conditions=conditions, qualifier=qualifier)

    electrolyte_k = '1 M LiTFSI + 1 wt% LiNO3; DOL:DME 1:1 v/v'
    ionic_conditions = dict(electrolyte=electrolyte_k, temperature_c=None,
                            temperature_note='Not specified for this EIS measurement in the extracted text.', cell='Al symmetric EIS')
    kim_cases = [('PP', 'none', 0, .43), ('BNNT-PP-0.3', 'raw_BNNT', .3, .71),
                 ('BNNT-PP-1.0', 'raw_BNNT', 1., None), ('p-BNNT-PP-0.3', 'purified_BNNT', .3, None),
                 ('p-BNNT-PP-0.5', 'purified_BNNT', .5, .84), ('p-BNNT-PP-1.0', 'purified_BNNT', 1., None),
                 ('p-BNNT-PP-2.0', 'purified_BNNT', 2., None), ('p-BNNT-PP-3.0', 'purified_BNNT', 3., None)]
    for name, form, loading, conductivity in kim_cases:
        inputs = dict(substrate='PP', bn_form=form, loading_mg_cm2=loading, binder='PVDF' if loading else 'none',
                      bn_binder_ratio=4. if loading else None, solvent='NMP' if loading else 'none',
                      dry_temperature_c=50. if loading else None, dry_hours=24. if loading else None,
                      sonication_hours=1. if loading else None, stirring='overnight' if loading else None,
                      electrolyte=electrolyte_k, test_temperature_c=None)
        observations = {}
        if conductivity is not None:
            observations['ionic_conductivity'] = obs(conductivity, 'mS/cm', k_ionic, ionic_conditions)
        add('kim_2022', name, 'PP_BNNT_LiTFSI_DOL_DME', inputs, observations,
            [k_recipe, k_electrolyte, k_ionic, k_diffusion, k_pair],
            'Loading unit corroborated in the diffusion paragraph; unlisted curve values are not extracted. '
            'The conclusion reports 0.50 mS/cm without explicit sample identity; that value is quarantined.',
            'test' if name == 'p-BNNT-PP-0.5' else 'train' if conductivity is not None else 'context')

    for name, gap, thickness in [('CA',0,73),('CA@BN-50',50,88),('CA@BN-100',100,110),('CA@BN-200',200,129),('PP-control',None,None),('CA@BN-3:1-failure',None,None)]:
        failure = 'failure' in name
        coated = name not in ['CA','PP-control']
        inputs = dict(substrate='PP' if name=='PP-control' else 'calcium_alginate',
                      bn_form='BN_nanopowder' if coated else 'none', binder='PVDF' if coated else 'none',
                      bn_binder_ratio=3. if failure else 4. if coated else None,
                      ratio_basis='Reported powder ratio; mass basis not explicitly stated.',
                      solvent='DMF' if coated else 'water' if name=='CA' else 'none',
                      applicator_gap_um=gap, dry_temperature_c=60. if coated else None,
                      dry_hours=None, coating_sides=None,
                      electrolyte='1 M LiPF6; EC:DMC:DEC 1:1:1 v/v/v', test_temperature_c=65.)
        observations = {}
        if thickness is not None:
            observations['final_thickness'] = obs(thickness,'um',t_thickness,dict(state='after_fabrication'))
        if name in ['CA@BN-100','PP-control']:
            observations['ionic_conductivity'] = obs(2.8 if name=='CA@BN-100' else 1.28, 'mS/cm', t_ionic,
                        dict(temperature_c=65.,electrolyte=inputs['electrolyte'],cell='SS symmetric EIS'))
        if failure:
            observations['preparation_outcome'] = obs('failed_pore_clogging','category',t_recipe,
                        dict(applicator_gap_um=None, coating_sides=None))
        add('tian_2024',name,'CA_BN_LiPF6_carbonates',inputs,observations,
            [t_recipe,t_ca,t_thickness,t_electrolyte,t_ionic],
            'Applicator setting differs from measured final thickness. Coating side and drying duration are unspecified; '
            'failure recipe retained with unspecified geometry. Transference number conflict is quarantined.')

    for sample, bn, value in [('PEO-PVDF',0,.06),('BN-PEO-PVDF',12,.2)]:
        inputs=dict(substrate='solid_PEO_PVDF',bn_form='BN_flakes' if bn else 'none', bn_weight_pct=bn,
                    loading_basis='Reported wt%; denominator not specified' if bn else 'no BN added',
                    polymer_salt_ratio='PEO:PVDF:LiTFSI 3:1:1', solvent='NMP',stirring_hours=12.,
                    dry_temperature_c=80.,dry_hours=5.,electrolyte='solid PEO-PVDF-LiTFSI',test_temperature_c=70.)
        add('yin_2020',sample,'solid_PEO_PVDF_LiTFSI',inputs,
            {'ionic_conductivity':obs(value,'mS/cm',y_ionic,dict(temperature_c=70.,state='after_temperature_step_stabilization'), 'approximate; converted from S/cm')},
            [y_recipe,y_ionic], 'Solid electrolyte: excluded from liquid-wetted separator prediction.')

    for name,bn,thickness,angle,sd in [('HCNF',0,25,28.45,3.34),('HBNT-05',.5,35,50.23,3.14),('HBNT-10',1.,35,68.54,1.55),('PE-control',None,None,None,None)]:
        inputs=dict(substrate='PE' if bn is None else 'cellulose',bn_form='BNNT' if bn else 'none',
                    bn_weight_pct=bn, loading_basis='Reported wt%; denominator not specified' if bn else None,
                    solvent='IPA:water 95:5 v/v' if bn is not None else 'none',stirring_hours=2. if bn is not None else None,
                    dry_temperature_c=80. if bn is not None else None,dry_hours=48. if bn is not None else None,
                    electrolyte='1 M LiPF6 + 5 vol% FEC; EC:EMC 3:7 v/v',test_temperature_c=None)
        observations={}
        if thickness is not None:
            observations['final_thickness']=obs(thickness,'um','hong_2026:table2',dict(state='after_fabrication'))
            observations['water_contact_angle']=obs(angle,'degree','hong_2026:table2',dict(liquid='water',reported_plus_minus=sd))
        add('hong_2026',name,'CNF_BNNT_LiPF6_carbonates',inputs,observations,[h_recipe,h_cell,'hong_2026:table2',h_capacity],
            'Water contact angle is not electrolyte contact angle. Published C-rate degree symbols are typographical; no temperature is inferred.')

    # Screened leads are not silently turned into training samples.
    leads = [
        ('rodriguez_2020','10.1016/j.jcis.2020.09.009','Engineered heat dissipation and current distribution boron nitride-graphene layer coated on polypropylene separator for high performance lithium metal battery','https://pubmed.ncbi.nlm.nih.gov/33010580/','abstract_only','PP coating; full methods needed; DOI verified through Europe PMC PMID metadata'),
        ('rahman_2019','10.1016/j.ensm.2019.03.027','High temperature and high rate lithium-ion batteries with boron nitride nanotubes coated polypropylene separators','https://doi.org/10.1016/j.ensm.2019.03.027','abstract_only','PP coating; seek permitted manuscript/supplement'),
        ('sheng_2021','10.1021/acs.nanolett.1c03106','Stabilized Solid Electrolyte Interphase Induced by Ultrathin Boron Nitride Membranes for Safe Lithium Metal Batteries','https://doi.org/10.1021/acs.nanolett.1c03106','abstract_only','PP/PI coating; methods not extracted'),
        ('fan_2019','10.1021/acsaem.8b02205','Repelling Polysulfide Ions by Boron Nitride Nanosheet Coated Separators in Lithium-Sulfur Batteries','https://pubs.acs.org/doi/10.1021/acsaem.8b02205','abstract_only','Purchased full text; seek author manuscript'),
        ('tin_bn_2025',None,'Dual-functional TiN/BN@separator for enhanced electrochemical performance and safety of lithium-ion batteries','https://www.sciencedirect.com/science/article/pii/S2352152X25023217','abstract_only','0.466 mS/cm lacks full recipe/conditions here; not extracted'),
        ('chen_2017','10.1371/journal.pone.0170523','Polydopamine-functionalized boron nitride in polypropylene composites','https://pmc.ncbi.nlm.nih.gov/articles/PMC5249180/','prior_context_lead','Bulk polymer thermal conductivity; not separator labels'),
        ('bouville_2014','10.1111/jace.12653','Boron nitride aqueous dispersion with cellulose','https://arxiv.org/abs/1710.04239','prior_context_lead','Dispersion context; not separator labels'),
        ('sleiti_2021','10.1016/j.dib.2021.106881','PAO/hBN measured-viscosity dataset','https://pmc.ncbi.nlm.nih.gov/articles/PMC7907776/','prior_context_lead','Oil rheology; incompatible endpoint/system'),
    ]
    for sid,doi,title,url,access,limit in leads:
        sources.append(dict(source_id=sid,doi=doi,title=title,url=url,full_text_url=None,
                            accessed_utc='2026-09-27',access=access,license='not_verified',
                            use='metadata_links_only',scope='screened_lead',limitation=limit))

    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(schema_version=1,dataset_version='separator-pilot-v1',sources=sources,
                 evidence=evidence,records=records,
                 quarantined=[dict(source_id='kim_2022',evidence_ids=[k_capacity],reason='1197/1675*100 = 71.4627%, not the reported 72.6%; retain both, exclude derived claim.'),
                              dict(source_id='kim_2022',evidence_ids=[k_failure],reason='0.50 mS/cm has ambiguous sample identity; excluded from numeric labels.'),
                              dict(source_id='tian_2024',evidence_ids=[t_conflict,t_conclusion],reason='Maximum 0.25 statement conflicts with 0.50/0.55; transference numbers excluded.')])
    (OUT/'dataset.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'source_inventory.csv').open('w',newline='') as f:
        fields=['source_id','doi','title','url','access','license','use','scope','limitation']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n'); writer.writeheader();writer.writerows(sources)
    with (OUT/'pilot_records.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['record_id','source_id','sample','cohort','split','inputs_json','observations_json','evidence_ids','notes'])
        for r in records:
            writer.writerow([r['record_id'],r['source_id'],r['sample'],r['cohort'],r['split'],json.dumps(r['inputs'],ensure_ascii=False),json.dumps(r['observations'],ensure_ascii=False),'|'.join(r['evidence_ids']),r['notes']])
    print(json.dumps(dict(sources=len(sources),primary_studies=4,records=len(records),observations=sum(len(r['observations']) for r in records))))


if __name__ == '__main__':
    main()
