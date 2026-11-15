# UX y accesibilidad

Objetivo: WCAG 2.2 AA, con principios de ISO 9241-210/11/112 y heurísticas de Nielsen como referencias de diseño. **No existe certificación ISO ni declaración de conformidad exhaustiva.**

Tokens propios en `src/styles/global.css`: carbón/verde en tema oscuro y superficies claras con acentos de contraste en tema claro. La Home es una superficie de investigación, no marketing. Sidebar compacta, jerarquía tipográfica, bordes discretos, visualizaciones dominantes y tablas con scroll local.

Semántica: landmarks, títulos, tablas con caption, listas de definición, labels nativos, estados textuales y aria-pressed para selección. Los botones shadcn componen Radix cuando es necesario; los selects nativos conservan comportamiento de teclado. Skip link y foco visible; el foco de ruta no desplaza automáticamente el viewport. La navegación móvil se oculta también para interacción cuando está cerrada y puede cerrarse con Escape.

Los gráficos disponen de alternativas: botones por clúster, selector de contenido, detalles y tablas de datos para curvas/histogramas. AFTER SPLIT usa forma además de color: círculo/train, cuadrado/validation, rombo/test; noise mantiene triángulo y etiqueta −1. La procedencia y asignación se pueden leer en detalles sin interpretar colores.

Desktop-first: comprobaciones a 1024, 1440 y 1920 px. En móvil la navegación se despliega, los paneles apilan y las tablas desbordan solo en su contenedor. ECharts sin animación; media query de reduced motion elimina transiciones. Tema claro/oscuro como preferencia local.

Axe prueba pantallas principales en ambos temas; Playwright comprueba navegación responsive y funcionalidades. Las pruebas automáticas no reemplazan auditoría manual exhaustiva, screen readers, zoom alto ni evaluación con investigadores reales. Las imágenes no están conectadas: futuros adapters deben proporcionar descripciones, overlays alineados y estados de error/acceso.

## Progressive disclosure y UX v2

Pregunta → visualización → resultado conciso → detalles. M02 tiene cuatro secciones; no se agregan pestañas por familia de metadata. Un resumen presenta hasta cinco métricas. IDs de espacios, configuración, semillas, pooling, integridad y métricas completas viven en drawers; el ID de un contenido seleccionado permanece visible porque identifica el objeto inspeccionado.

`TechnicalDetailsDrawer` usa Dialog de Radix: trigger semántico, título/descripción, foco atrapado, Escape y retorno al trigger. El contenido pesado se monta solo al abrir. El scatter comparte selección con detalle, galería y grilla; la tabla completa se abre con un summary accesible por teclado y pagina 25 filas. Las secuencias paginan 20 intervalos y linkage 12 pares.

Candidatos: patrón discontinuo y signo de interrogación. Revisión/aceptación/rechazo/ambigüedad mantienen texto, además de estilo; nunca solo color. Boundary zone no se dibuja como corte exacto. Sequence instance se distingue de sequence candidate. Los cosenos de encoders no se promedian. Un reviewed/supported no se presenta como ground truth ni link confirmado.

La Home usa enlaces nativos y una bifurcación visual ligera sin introducir dependencias de grafos. Las transiciones responden a interacción y respetan reduced motion. Los espacios de galería declaran imágenes no disponibles; no se copian frames privados para decorar el sitio.
