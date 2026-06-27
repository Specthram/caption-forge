/** Root: routes the active view (Zustand) inside the app shell. */

import { AppShell } from "./components/templates/AppShell";
import { CaptionView } from "./pages/CaptionView";
import { DatasetsView } from "./pages/DatasetsView";
import { LibrariesView } from "./pages/LibrariesView";
import { MediaView } from "./pages/MediaView";
import { SettingsView } from "./pages/SettingsView";
import { SystemView } from "./pages/SystemView";
import { TagsView } from "./pages/TagsView";
import { useUiStore } from "./store/uiStore";

export default function App() {
  const view = useUiStore((state) => state.view);
  return (
    <AppShell>
      {view === "caption" && <CaptionView />}
      {view === "datasets" && <DatasetsView />}
      {view === "media" && <MediaView />}
      {view === "tags" && <TagsView />}
      {view === "libraries" && <LibrariesView />}
      {view === "settings" && <SettingsView />}
      {view === "system" && <SystemView />}
    </AppShell>
  );
}
