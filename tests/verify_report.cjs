const fs=require('fs');
const vm=require('vm');
const assert=require('assert');
const file=process.argv[2];
const content=fs.readFileSync(file,'utf8');
assert(!/__[A-Z_]+__/.test(content),'Unfilled report token');
const json=JSON.parse(content.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const script=content.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements=new Map();
function element(id){
 if(!elements.has(id))elements.set(id,{
  id,value:id==='holding'?'0.01':id==='shortage'?'1':'',textContent:id==='report-data'?JSON.stringify(json):'',
  parentElement:{clientWidth:1050},attrs:{},events:{},_html:'',
  get innerHTML(){return this._html},set innerHTML(v){this._html=v;if(['family','model','origin'].includes(id)&&!this.value)this.value=v.match(/value="([^"]+)"/)[1]},
  setAttribute(k,v){this.attrs[k]=v},addEventListener(k,v){this.events[k]=v},
  querySelector(){return element(id+'-tbody')}
 });
 return elements.get(id)
}
const context=vm.createContext({document:{getElementById:element},window:{addEventListener(){}},console});
vm.runInContext(script,context);
assert.equal((element('metrics').innerHTML.match(/<tr/g)||[]).length,json.models.length);
assert.equal((element('family-metrics').innerHTML.match(/<tr/g)||[]).length,4);
if(json.comparisonBoosting){
 const matchedBoosting=json.metrics.find(r=>r.model===json.comparisonBoosting);
 assert(content.includes(`<strong>${(100*matchedBoosting.wape).toFixed(2)}%</strong>`),'Headline does not match current boosting results');
 assert(!content.includes('Almost tied with TimesFM'),'Stale initial-comparison conclusion');
 for(const family of json.families){
  const score=json.byFamily.find(r=>r.family===family&&r.model===json.comparisonBoosting);
  assert(element('family-metrics').innerHTML.includes((100*score.wape).toFixed(2)+'%'),'Family table uses another baseline');
 }
}
for(const family of json.families)for(const model of json.models)for(const origin of json.origins){
 element('family').value=family;element('model').value=model;element('origin').value=origin;
 element('family').events.change();
 assert.equal((element('forecast-table-tbody').innerHTML.match(/<tr/g)||[]).length,14);
 assert(!/NaN|undefined/.test(element('forecast').innerHTML));
 const expected={'BEVERAGES':'Beverages','BREAD/BAKERY':'Bread / bakery','DAIRY':'Dairy','GROCERY I':'Grocery I'}[family];
 assert(element('forecast-caption').textContent.includes(expected));
}
element('forecast').parentElement.clientWidth=360;
element('family').events.change();
assert(!/NaN|undefined/.test(element('forecast').innerHTML));
element('holding').value='0.05';element('holding').events.input();
assert(element('inventory').innerHTML.startsWith('<tr class="best"><td>TimesFM: groups together + calendar'));
element('holding').value='0.01';element('shortage').value='1';element('shortage').events.input();
assert(element('inventory').innerHTML.startsWith('<tr class="best"><td>TimesFM: groups together</td>'));
assert.equal((element('inventory').innerHTML.match(/<tr/g)||[]).length,json.models.length);
console.log(JSON.stringify({reportDataBytes:JSON.stringify(json).length,forecastRows:json.forecasts.length,filterCombinationsChecked:json.families.length*json.models.length*json.origins.length,sliderRankingChecks:2,scriptRuntime:'passed',browserRendering:'unavailable: in-app browser cannot reach local host'}));
