function asText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function factEntries(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value).filter(([key]) => key);
}

function listItems(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(itemText).filter((item) => item.trim());
}

function itemText(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return asText(
      value.label ?? value.text ?? value.title ?? value.value ?? value.name,
    );
  }
  return asText(value);
}

function tableColumns(table) {
  if (!table || typeof table !== "object" || !Array.isArray(table.columns)) {
    return [];
  }
  return table.columns.map(asText).filter((column) => column.trim());
}

function tableRows(table) {
  if (!table || typeof table !== "object" || !Array.isArray(table.rows)) {
    return [];
  }
  return table.rows;
}

function cellValue(row, column, index) {
  if (Array.isArray(row)) {
    return asText(row[index]);
  }
  if (row && typeof row === "object") {
    return asText(row[column]);
  }
  return index === 0 ? asText(row) : "";
}

function actionItems(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((action) => ({
      label: asText(action?.label).trim(),
      prompt: asText(action?.prompt).trim(),
    }))
    .filter((action) => action.label && action.prompt);
}

function uniqueActions(...groups) {
  const seen = new Set();
  const merged = [];
  groups.flat().forEach((action) => {
    const key = `${action.label}\n${action.prompt}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(action);
    }
  });
  return merged;
}

export default function GeneratedPanel() {
  const title = asText(props.title).trim() || "Generated panel";
  const summary = asText(props.summary).trim();
  const facts = factEntries(props.facts);
  const items = listItems(props.items);
  const columns = tableColumns(props.table);
  const rows = tableRows(props.table);
  const actions = uniqueActions(
    actionItems(props.actions),
    actionItems(props.items),
  );

  return (
    <section className="mt-3 w-full rounded-md border border-border bg-card text-card-foreground shadow-sm">
      <div className="border-b border-border px-4 py-3">
        <h3 className="text-base font-semibold leading-6">{title}</h3>
        {summary ? (
          <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-muted-foreground">
            {summary}
          </p>
        ) : null}
      </div>

      {facts.length ? (
        <dl className="grid grid-cols-1 border-b border-border sm:grid-cols-2">
          {facts.map(([key, value]) => (
            <div key={key} className="border-border px-4 py-3 sm:border-r">
              <dt className="text-xs font-medium uppercase tracking-normal text-muted-foreground">
                {key}
              </dt>
              <dd className="mt-1 text-sm leading-6">{asText(value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {items.length ? (
        <ul className="space-y-2 border-b border-border px-4 py-3 text-sm leading-6">
          {items.map((item, index) => (
            <li key={`${index}-${item}`} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-primary" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {columns.length && rows.length ? (
        <div className="overflow-x-auto border-b border-border">
          <table className="w-full min-w-max text-left text-sm">
            <thead className="bg-muted/60 text-xs uppercase text-muted-foreground">
              <tr>
                {columns.map((column) => (
                  <th key={column} className="px-4 py-2 font-medium">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="border-t border-border">
                  {columns.map((column, columnIndex) => (
                    <td key={column} className="px-4 py-2 align-top">
                      {cellValue(row, column, columnIndex)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {actions.length ? (
        <div className="flex flex-wrap gap-2 px-4 py-3">
          {actions.map((action) => (
            <button
              key={`${action.label}-${action.prompt}`}
              type="button"
              className="rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted"
              onClick={() => sendUserMessage(action.prompt)}
            >
              {action.label}
            </button>
          ))}
        </div>
      ) : null}
    </section>
  );
}
