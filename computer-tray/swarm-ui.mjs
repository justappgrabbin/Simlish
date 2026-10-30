/**
 * Executable formations in the EXISTING SynthAI Computer Systems view.
 * This is a UI binding, not a second OS or separate runtime.
 */
const AUTOLING_ID = 'klein:autoling';
const DISEMINER_ID = 'klein:diseminer';
const COMPOSED_ID = 'klein:autoling+diseminer';
const byId = id => document.getElementById(id);

const graphFor = operation => ({
  nodes: [{ id: 'input' }, { id: 'result' }],
  edges: [{ from: 'input', to: 'result', functionId: operation }],
});

export function mountSwarmComputerUI({ computer }) {
  const formations = () => computer.formations.list();
  const find = id => formations().find(f => f.id === id);
  const message = (text, okay = false) => {
    const element = byId('formationMessage');
    element.textContent = text;
    element.className = 'message ' + (okay ? 'good' : 'bad');
  };

  async function render(prefer = byId('formationPicker').value) {
    const list = formations();
    const picker = byId('formationPicker');
    const options = list.filter(f => f.id !== DISEMINER_ID);
    const selected = options.some(f => f.id === prefer) ? prefer
      : options.some(f => f.id === COMPOSED_ID) ? COMPOSED_ID
        : (options[0]?.id ?? '');
    picker.replaceChildren();
    if (!options.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'Assemble a formation first';
      picker.append(option);
    }
    for (const f of options) {
      const option = document.createElement('option');
      option.value = f.id;
      option.textContent = f.id;
      picker.append(option);
    }
    picker.value = selected;
    byId('formationStatus').textContent = String(list.length) + ' executable';
    const container = byId('formationList');
    container.replaceChildren();
    if (!list.length) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = 'No formations have been mounted yet.';
      container.append(empty);
    }
    for (const f of list) {
      const revision = f.revisions.find(r => r.id === f.activeRevision);
      const row = document.createElement('div');
      row.className = 'formation-row';
      const label = document.createElement('strong');
      label.textContent = f.id;
      const count = document.createElement('span');
      count.className = 'state';
      count.textContent = String(f.revisions.length) + ' revision(s)';
      const operations = document.createElement('small');
      operations.textContent = (revision?.graph?.edges ?? [])
        .map(edge => edge.functionId.replace('formation.revision:', '')).join(' → ');
      const address = document.createElement('small');
      address.textContent = 'Gate ' + f.address.gate + ' · Active ' + f.activeRevision;
      row.append(label, count, operations, address);
      container.append(row);
    }
    if (computer.nativeKlein) {
      try {
        const snapshot = await computer.nativeKlein.snapshot();
        byId('formationMemory').textContent =
          String(snapshot.rules.length) + ' AUTOLING grammar rules · ' +
          String(snapshot.documentCount) + ' native lexical records · ' +
          String(snapshot.terms.length) + ' observed terms' +
          (snapshot.terms.length ? ': ' + snapshot.terms.slice(0, 14).join(', ') : '.');
      } catch (error) {
        byId('formationMemory').textContent = 'Native memory unavailable: ' + error.message;
      }
    } else byId('formationMemory').textContent = 'Native Klein tools are not mounted.';
  }

  async function assemble() {
    const donor = computer.nativeKlein;
    if (!donor) return message('Native Klein donor is unavailable in this Computer');
    try {
      // Addresses, behavior and recognition rules come from the installed donor.
      if (!find(AUTOLING_ID)) {
        await computer.createFormation({
          id: AUTOLING_ID,
          address: donor.autoLing.address,
          purpose: 'native-autoling-grammar-recognition',
          graph: graphFor('klein.autoling.recognize'),
        });
      }
      if (!find(DISEMINER_ID)) {
        await computer.createFormation({
          id: DISEMINER_ID,
          address: donor.diseminer.address,
          purpose: 'native-diseminer-lexical-memory',
          graph: graphFor('klein.diseminer.ingest-recognized'),
        });
      }
      if (!find(COMPOSED_ID)) {
        await computer.composeFormations({
          id: COMPOSED_ID, sourceId: AUTOLING_ID, targetId: DISEMINER_ID,
          purpose: 'native Klein recognition plus lexical memory',
        });
      }
      await render(COMPOSED_ID);
      message('Native AUTOLING + Diseminer composition is mounted and executable.', true);
    } catch (error) {
      message('ATO assembly failed: ' + error.message);
      await render();
    }
  }

  async function execute() {
    const id = byId('formationPicker').value;
    const text = byId('formationText').value.trim();
    if (!id) return message('Select an executable formation first.');
    if (!text) return message('A sentence is required for the Klein tool.');
    try {
      const receipt = await computer.executeFormation(id, { text });
      const output = Object.values(receipt.outputs)[0];
      let summary;
      if (output?.document) {
        summary = 'Recognized ' + output.relation + ' · ' +
          output.document.tokens.join(' / ') + ' · Memory record ' + output.document.id;
      } else if (output?.matches) {
        summary = output.matches.length
          ? 'AUTOLING recognized ' + output.matches.map(m =>
            m.relation + ' (' + Object.values(m.bindings).join(' / ') + ')').join('; ')
          : 'AUTOLING did not recognize a relation in this sentence.';
      } else if (output?.ok === false) {
        summary = output.reason;
      } else {
        summary = 'The formation executed; output is retained in Computer history.';
      }
      byId('formationOutput').textContent = summary;
      message('Executed ' + receipt.revisionId + ' using local ATO.', true);
      await render(id);
    } catch (error) {
      byId('formationOutput').textContent = 'Execution failed: ' + error.message;
      message(error.message);
    }
  }

  async function teachRule() {
    if (!computer.nativeKlein) {
      byId('ruleMessage').textContent = 'Native AUTOLING is not available.';
      return;
    }
    const pattern = byId('rulePattern').value.trim().split(/\s+/).filter(Boolean);
    const rule = {
      id: byId('ruleId').value.trim(),
      relation: byId('ruleRelation').value.trim(),
      pattern,
      template: byId('ruleTemplate').value.trim(),
    };
    try {
      const learned = await computer.nativeKlein.learnRule(rule);
      byId('ruleMessage').textContent =
        'Native AUTOLING learned ' + learned.relation + ' (' + learned.id + '). This rule is preserved in Computer history.';
      await render();
    } catch (error) {
      byId('ruleMessage').textContent = 'AUTOLING rule not accepted: ' + error.message;
    }
  }

  byId('teachAutoling').addEventListener('click', teachRule);
  byId('assembleKlein').addEventListener('click', assemble);
  byId('refreshFormations').addEventListener('click', () => void render());
  byId('runFormation').addEventListener('click', execute);
  return { render };
}
