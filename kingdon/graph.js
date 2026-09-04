const ganja_source = sessionStorage.ganja_source ||
    await fetch("https://enki.ws/ganja.js/ganja.js")
          .then(x => x.text());

if (!sessionStorage.ganja_source) {sessionStorage.ganja_source = ganja_source};

const Algebra = (() => {
    const ctx = {};
    (new Function('const define=1;' + ganja_source)).apply(ctx);
    return ctx.Algebra;
})();

function render({ model, el }) {
    var canvas = null;

    function createCanvas() {
        if (canvas && canvas.parentNode) {
            canvas.parentNode.removeChild(canvas);
        }

        canvas = Algebra({metric: model.get('signature'), basis: model.get('basis'), graded: model.get('graded')}).inline((model)=>{
            // Define constants
            var key2idx = model.get('key2idx');
            var draggable_points_idxs = model.get('draggable_points_idxs');
            var types = model.get('types') || {};
            var options = model.get('options');
            var layouts = Object.fromEntries(
                Object.entries(types).map(([name, L])=>[name, Object.entries(L).map(([k, v])=>[k|0, v])])
            );

            // Define helper functions.
            function grade(key) {
                var count = 0;
                while (key) {
                    count += key & 1;
                    key >>= 1;
                }
                return count;
            }
            var layout = (o)=>layouts[o['type']] || [];
            var TYPED_ARRAYS = {
                '|b1': Uint8Array,   // numpy bool is one byte per element; js has no boolean typed array
                '|u1': Uint8Array,
                '|i1': Int8Array,
                '<u2': Uint16Array,
                '<i2': Int16Array,
                '<u4': Uint32Array,
                '<i4': Int32Array,
                '<f4': Float32Array,
                '<f8': Float64Array,
            };
            var toArray = (o)=>{
                /* read a {dtype, shape, buffer} object into an array shaped like the multivector.

                   Python sends a struct of arrays: one array of coefficients per blade, so
                   the blade axis comes first and the shape is [nkeys, ...mv.shape]. We turn
                   that inside out into an array of structs, so the innermost entries are
                   typed arrays of the blade coefficients of one multivector, each ready to
                   hand to toElement. */
                var TypedArray = TYPED_ARRAYS[o['dtype']];
                if (TypedArray === undefined) throw new Error("toArray: unsupported dtype " + o['dtype']);
                var dv = o['buffer'];  // Retrieve the DataView
                var flat = new TypedArray(dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength));

                var nkeys = o['shape'][0];
                var stride = flat.length / nkeys;  // number of multivectors in the blob
                var structs = [];
                for (var i = 0; i < stride; i++) {
                    // The coefficients of multivector i sit a stride apart in the blob.
                    var values = new TypedArray(nkeys);
                    for (var j = 0; j < nkeys; j++) values[j] = flat[j * stride + i];
                    structs.push(values);
                }

                var nest = (a, shape)=>{
                    if (shape.length === 0) return a[0];
                    if (shape.length === 1) return a;
                    var step = a.length / shape[0];
                    var out = [];
                    for (var i = 0; i < shape[0]; i++) out.push(nest(a.slice(i * step, (i + 1) * step), shape.slice(1)));
                    return out;
                }
                return nest(structs, o['shape'].slice(1));
            }

            var toElement = (o, _values=o['mv'])=>{
                /* convert object to Element */
                if ('grades' in o) {
                    var values = new Element();
                    o['grades'].forEach(g=>values[g] = []);
                    layout(o).forEach(([k, v])=>{var g = grade(k); values[g][key2idx[g][k]] = v});
                    o['keys'].forEach((k, j)=>{var g = grade(k); values[g][key2idx[g][k]] = _values[j]});
                    return values;
                }
                if ('keys' in o) {
                    var values = Array(Object.keys(key2idx).length).fill(0);
                    layout(o).forEach(([k, v])=>values[key2idx[k]] = v);
                    o['keys'].forEach((k, j)=>values[key2idx[k]] = _values[j]);
                    return new Element(values);
                }
                return new Element(_values);
            }
            var unpack = (o)=>{
                /* convert object to Element, or to an array of Elements shaped like the mv. */
                if (Array.isArray(o['mv'])) return toElement(o);
                var map = (x)=>Array.isArray(x)?x.map(map):toElement(o, x);
                return map(toArray(o['mv']));
            }
            var decode = x=>typeof x === 'object' && 'mv' in x?unpack(x):Array.isArray(x)?x.flatMap(spread):x;
            var spread = (x)=>{
                /* decode one item of a list, splicing a shaped mv into that list. Ganja reads
                   an array as one polygon, so shape (N,) has to become N points instead of one
                   N-gon, and shape (N, 3) N arrays of three points, which are N triangles.
                   A shaped mv decodes to an array of Elements, a scalar one to a single Element;
                   test for that and not for Array, because a graded Element is itself an Array. */
                var decoded = decode(x);
                return typeof x === 'object' && 'mv' in x && !(decoded instanceof Element)?decoded:[decoded];
            }
            var encode = x=>x instanceof Element?({mv:[...x]}):x?.map?x.map(encode):x;

            // Decode camera if provided.
            if (options?.camera && typeof options.camera === 'object' && 'mv' in options.camera) {
                options.camera = unpack(options.camera)
            }

            // Unregister previous change:subjects handler if it exists.
            if (model._subjectsHandler) {
                model.off("change:subjects", model._subjectsHandler);
                model._subjectsHandler = null;
            }

            if (options?.animate) {
                var graph_func = ()=>{
                    if (canvas?.value && draggable_points_idxs?.length) {
                        model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                        model.save_changes();
                    }
                    // Send an update request. This drives the event loop.
                    model.send({ type: "update_mvs" });
                    var subjects = decode(model.get('subjects'));
                    return [...subjects];
                }
            } else {
                var graph_func = ()=>{
                    if (canvas?.value && draggable_points_idxs?.length) {
                        model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                        model.save_changes();
                    }
                    var subjects = decode(model.get('subjects'));
                    return [...subjects];
                }

                // This ensures the remake is always called one last time to show the final position.
                model._subjectsHandler = ()=>{
                    if (canvas.remake) canvas = canvas.remake(0);
                    if (canvas.update) canvas.update(canvas.value);
                };
                model.on("change:subjects", model._subjectsHandler);
            }

            var canvas;
            canvas = this.graph(graph_func, options)
            return canvas;
        })(model)

        var style = model.get('options')?.style || {};
        for (var prop in style) {
            canvas.style[prop] = style[prop];
        }
        el.appendChild(canvas);
    }

    createCanvas();
    model.on("change:options", () => createCanvas());
}

export default { render };
