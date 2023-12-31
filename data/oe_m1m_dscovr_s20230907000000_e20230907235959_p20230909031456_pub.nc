CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230907000000_e20230907235959_p20230909031456_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-09-09T03:14:56.163Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-09-07T00:00:00.000Z   time_coverage_end         2023-09-07T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data           records_fill         �   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��@  �          @��H?޸R�}p���  ���\C���?޸R���R��\��z�C�8R                                    Bx���  �          @>�R>��
�\�G�� C��>��
�h�����C���                                    Bx��!�  �          @L�;aG�@ �����F{B���aG�?���1��|��B�\                                    Bx��02  �          @g�?@  ��\)�^�R�C��=?@  �����Tz�(�C��                                    Bx��>�  �          @�=q?B�\@*=q�U��I
=B��?B�\?���w��}Q�B��                                    Bx��M~  �          @�G�?
=@p���S33�#��B�G�?
=@7
=��33�Z�B��)                                    Bx��\$  �          @��>�ff@u��QG��!z�B�z�>�ff@;���33�XffB�k�                                    Bx��j�  �          @���?
=q@xQ��I���33B��?
=q@@���\)�Q��B���                                    Bx��yp              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ψ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ϖ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ϥb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ϴ   �          @��\?�R@%�Y���O�
B��
?�R?�Q��z�H��B�p�                                    Bx��®  T          @��H?
=q@333�QG��Cz�B�=q?
=q?�
=�vff�y�HB��
                                    Bx���T  �          @�z�\)@��o\)�j33B�
=�\)?������B�33                                    Bx����  �          @��H>�Q�@���b�\�^�\B��H>�Q�?�(���Q�B�k�                                    Bx���  �          @�z�>�z�@(��fff�^��B��3>�z�?�  ���\  B�z�                                    Bx���F  T          @�ff?(�@J=q�C33�-�B�\?(�@��n{�d33B��                                    Bx���  �          @��?B�\@:=q�>�R�2�
B��R?B�\@ff�e�hz�B�=q                                    Bx���  �          @|��?@  @:�H�"�\� z�B�#�?@  @p��J�H�V\)B��=                                    Bx��)8  �          @w�?�@=p��=q�  B�?�@��C33�R(�B�L�                                    Bx��7�  �          @U�>�ff@�
�33�1�\B��f>�ff?�
=�333�hp�B���                                    Bx��F�  �          @U�?.{?�\�*=q�Z�B��?.{?����AG�ffBf��                                    Bx��U*  �          @S33?^�R@�\�ff�:��B�?^�R?�33�1��m33Bh                                    Bx��c�  �          @Tz�+�@   ��
��HB�
=�+�?�
=�'
=�O�B�z�                                    Bx��rv  �          @O\)���
@ff���$�HBýq���
?�\�%�\B�Ǯ                                    Bx�Ё  �          @O\)���?�=q�G��<=qB�����?�(��)���j��C��                                    Bx�Џ�  �          @aG���\)@p���\)��
B�(���\)?���H�6�B���                                    Bx�Оh  �          @a녿�\?���z��'�HCQ��\?�  �-p��MffC�                                     Bx�Э  �          @a녾#�
?�
=�;��pG�B��{�#�
?h���P  Q�B�ff                                    Bx�л�  �          @`��>�(�?���7��`z�B�8R>�(�?�\)�O\)�B��)                                    Bx���Z  �          @^{��  ?��R�4z��Z�\B�p���  ?�p��N{aHB��                                    Bx���   �          @Z�H�k�@(����/�B��=�k�?��
�9���h��B�ff                                    Bx���  �          @Z�H�ٙ�@������=qB�(��ٙ�?�(��޸R���C�)                                    Bx���L  �          @S33���@G
=?8Q�AN�\B�zᾅ�@L(���\)��=qB�=q                                    Bx���  �          @'
=���R?\��G���\CT{���R?��Ϳ��3ffCz�                                    Bx���  �          @4z�޸R?�33������
=Cz�޸R?�G��޸R�\)C
                                    Bx��">  �          @&ff�ff?�z���&�\C8R�ff?�  �L�����RCQ�                                    Bx��0�  �          @�Ϳ�Q�>�녿5��C'���Q�>aG��J=q��ffC-k�                                    Bx��?�  �          @Q��
�H>��H�L�����HC'Q��
�H>�=q�fff����C,��                                    Bx��N0  �          @{���H?&ff��(����RC!�����H>��ÿ����\)C*�                                     Bx��\�  �          @   �Ǯ?\(���z��,=qC)�Ǯ>�
=��=q�B{C$��                                    Bx��k|  �          @
=q���?k��8Q����HC�\���?5�k���
=C\                                    Bx��z"  �          ?��ÿ�\?Y�������RB��
��\?J=q��33���B���                                    Bx�ш�  �          @�
��>.{>�(�A5�C/�=��>�=q>�p�AffC,��                                    Bx�їn  �          @G����?
=>���@�Q�C$�����?&ff>\)@l��C#)                                    Bx�Ѧ  �          ?�Q쿷
=?J=q��Q��EC#׿�
=?+��
=q���C�=                                    Bx�Ѵ�  �          ?��ÿ��?
=q�n{���C5ÿ��>�\)����-�\C%                                      Bx���`  �          @ �׿�33>�z�k���33C*���33=L�Ϳu��33C2=q                                    Bx���  �          @����\�u�L����ffC:����\��\)�\)�u�C;�                                    Bx���  �          @���
=�^�R>L��@��COQ��
=�J=q>��AG�CM.                                    Bx���R  �          ?��z�H��  �.{��Q�Ca�R�z�H��녾���s�Ces3                                    Bx����  �          @!녿�zῢ�\?k�A���CU����z῀  ?��HA�z�CO��                                    Bx���  �          @
=q��(��@  ?xQ�A�G�CK�R��(���?���B�\CC�q                                    Bx��D  �          @�Ϳ��ͿxQ�?��Ak33CO�H���ͿL��?E�A�Q�CKT{                                    Bx��)�  �          @Q�?���
=�����C33C�?����H=#�
?У�C��3                                    Bx��8�  �          @=q?�\)?u�W
=���\A�\)?�\)?8Q쿇�����A���                                    Bx��G6  �          @1�?��
�@  ����,Q�C���?��
��=q��33�	�C�޸                                    Bx��U�  �          @U?aG��L(��u��G�C���?aG��Fff?B�\AS33C��{                                    Bx��d�  �          @_\)?��U?z�A(�C��
?��E�?�z�A\C�=q                                    Bx��s(  �          @s�
�\�J=q?޸RA�G�C��þ\�'
=@\)B-Q�C��                                     Bx�ҁ�  �          @vff?0���b�\?��\A��RC��H?0���Fff@Q�Bp�C�C�                                    Bx�Ґt  �          @xQ�?�=q�]p�?���A��C��?�=q�@  @
=qB33C��                                    Bx�ҟ  �          @}p�?�z��]p�?�ffA�{C�  ?�z��@��@	��B�\C��R                                    Bx�ҭ�  �          @z�H?��R�W
=?˅A�\)C��?��R�5@��B  C���                                    Bx�Ҽf  T          @tz�?+��G�@ffB��C��?+��{@5�B?�C���                                    Bx���  �          @�  >�  �o\)?�A�Q�C��>�  �S�
@B C�                                      Bx��ٲ  �          @�\)>���k�?�(�A�=qC�7
>���B�\@6ffB+�C��                                     Bx���X  �          @���>�\)�tz�?�\A�=qC�
>�\)�N�R@,��B�HC�xR                                    Bx����  �          @��
>����~{?�\A��C�'�>����W�@.�RB��C���                                    Bx���  �          @�G�>#�
�|(�?�z�A�Q�C�&f>#�
�W�@'�B=qC�XR                                    Bx��J  �          @�>Ǯ�p��?�G�A�C��{>Ǯ�J�H@+�B�C�~�                                    Bx��"�  �          @��?�\�hQ�?�A�p�C���?�\�A�@.�RB&\)C���                                    Bx��1�  �          @�Q�?��g
=?�\)A��C�*=?��C�
@   BffC��                                    Bx��@<  �          @��?W
=�x��?�@�ffC��?W
=�g�?\A���C���                                    Bx��N�  �          @��?
=q���>�  @_\)C��?
=q�w
=?�ffA�  C�H                                    Bx��]�  �          @��\>�z��|��?W
=A@z�C��>�z��e?���A�33C�T{                                    Bx��l.  �          @��H>�\)�|(�?L��A8  C�\>�\)�e?�A�33C�AH                                    Bx��z�  T          @~{>����s33?:�HA.{C��q>����^{?ٙ�A�  C�C�                                    Bx�Ӊz  �          @w
=>����c�
?�Q�A���C��>����C33@z�Bz�C�{                                    Bx�Ә   �          @vff=��e�?���A��RC���=��E@G�B\)C�!H                                    Bx�Ӧ�  ]          @{����e�?��RA���C��R���C�
@Q�Bz�C���                                    Bx�ӵl  �          @�Q�>u�j�H?˅A��RC�޸>u�G�@   Bz�C�33                                    Bx���  �          @��?ٙ��<(�@��B=qC�\?ٙ��(�@EB@\)C��                                    Bx��Ҹ  �          @��?У��H��@�A���C���?У����@;�B3p�C��R                                    Bx���^  �          @���?˅�W
=?�A�C��f?˅�0��@(Q�B�\C��
                                    Bx���  �          @��?��H�U�?��RA�G�C��q?��H�+�@333B*{C�P�                                    Bx����  �          @���?�z��Y��?��A���C�Ff?�z��1G�@.{B$��C�z�                                    Bx��P  �          @���?��U�@ ��A�{C���?��+�@4z�B+��C��)                                    Bx���  �          @�z�?�p��Q�?У�A��
C��?�p��.�R@��B�C�0�                                    Bx��*�  T          @�33?�33�`��?��A�p�C��R?�33�A�@��B(�C�H�                                    Bx��9B  T          @�G�?�p��b�\?\(�AD  C�?�p��K�?�\AиRC�E                                    Bx��G�  �          @��H?�(��b�\?�G�A�\)C�O\?�(��Dz�@
=qB��C��3                                    Bx��V�  �          @���?�\)�aG�?�
=A��C���?�\)�@  @z�B\)C�:�                                    Bx��e4  �          @�  ?���U?��A��\C�
?���7
=@(�B�\C��{                                    Bx��s�  �          @x��?����U?�ffA��C�� ?����7
=@
=qB�\C�p�                                    Bx�Ԃ�  �          @x��>�{�Tz�?��HA���C��{>�{�*�H@1�B7��C���                                    Bx�ԑ&  �          @z�H?   �c�
?�p�A�33C���?   �A�@��B�RC��\                                    Bx�ԟ�  �          @x��?u�hQ�?uAeC�j=?u�N�R?�z�A�{C�G�                                    Bx�Ԯr  �          @�  ?fff�qG�?uA^=qC���?fff�W�?���A�G�C�~�                                    Bx�Խ  �          @���>���w�?Tz�A@(�C�f>���`  ?���A݅C�XR                                    Bx��˾  �          @qG�?}p��J=q?��
A�p�C��{?}p��#�
@$z�B,z�C��
                                    Bx���d  �          @qG�?G��S�
?�\)A˅C���?G��/\)@p�B#{C��R                                    Bx���
  T          @n{?z�H�C33?��A�  C��f?z�H��H@)��B5�RC�f                                    Bx����  �          @k�?}p��1�@(�BG�C��3?}p��z�@7
=BM�C��=                                    Bx��V  �          @|��?����R�\?��A�C��?����)��@-p�B-��C�                                    Bx���  �          @��?�\)�j=q?�\)A�p�C��f?�\)�H��@z�BQ�C��3                                    Bx��#�  �          @�(�?^�R�s33?��A�(�C�q�?^�R�R�\@33B(�C�e                                    Bx��2H  �          @�G�?^�R�aG�?���A�ffC��\?^�R�<(�@ ��B
=C�:�                                    Bx��@�  "          @���?�(��\)?�G�AX��C���?�(��c�
@z�A�z�C�y�                                    Bx��O�  �          @�Q�?����g
=?���A��\C�f?����J=q@�A�Q�C�S3                                    Bx��^:  �          @��?�33�q�?�
=A�
=C�~�?�33�S33@�A�p�C��H                                    Bx��l�  �          @\)<����@I��BP��C�N<���p�@l��B��\C��H                                    Bx��{�  �          @\)>k��;�@,��B*=qC�H�>k���
@Y��Bj�\C�<)                                    Bx�Պ,  �          @}p�>���G�@�Bz�C�Z�>���z�@Mp�BW�C�*=                                    Bx�՘�  T          @���>�=q�K�@��B��C�ff>�=q��@O\)BVz�C�7
                                    Bx�էx  �          @~�R>�p��J�H@Q�B  C�\)>�p��Q�@J�HBS\)C�w
                                    Bx�ն  �          @xQ�>��H�Fff@�\BQ�C���>��H��@C�
BPQ�C���                                    Bx����  �          @vff?
=�A�@33B�\C�~�?
=�G�@C�
BR(�C�G�                                    Bx���j  �          @s�
?#�
�E@Q�B(�C�Ф?#�
��@:=qBG�RC��                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   Z          @~�R>u�e?�Q�A���C���>u�>{@(Q�B%C�O\                                    Bx���\  �          @~{?O\)�;�@��B�C��3?O\)�Q�@HQ�BX
=C�`                                     Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   Z          @�G�?�z��G
=@
=qBz�C�9�?�z��
=@<��B<  C�k�                                    Bx��+N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��W@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��e�   Z          @�  ?�(��g
=?k�AU�C�?�(��L(�?�A���C�S3                                    Bx��t�  T          @z�H@��<��?�=qA��C�=q@��(�@
=Bp�C��q                                   Bx�փ2  T          @~�R@ ���Dz�?��A�Q�C��)@ ���   @
=BQ�C�j=                                    Bx�֑�  
�          @�Q�@�\�Dz�?��A���C�˅@�\�   @
=B��C��q                                    Bx�֠~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�֯$   Z          @j=q?�p����H@@  B]z�C��?�p���@S33B�#�C�%                                    Bx�ֽ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���p   Z          @aG�?�{��(�@\)BffC���?�{���R@.{BJ=qC�5�                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   Z          @z=q@{�C33?}p�Al  C�f@{�(Q�?�A�RC��                                    Bx���b  "          @��@ff�Mp�?(��A�C��@ff�8Q�?���A�33C���                                    Bx��  �          @�Q�@G��O\)    =#�
C�~�@G��Fff?s33A]�C��                                    Bx���  T          @j�H?��A녿��C��\?��Dz�>�{@��C�p�                                    Bx��$T  "          @W���
=����3�
�c{C�w
��
=�!��
=q� =qC�XR                                    Bx��2�  �          @Y�����׿5�6ff�u��CO.���׿�G�� ���Kz�Cc�f                                    Bx��A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��PF   Z          @W�>���E���=q����C�g�>���U��p���Q�C�8R                                    Bx��^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��m�   Z          @g
=?�Q��R�\�^�R�^�HC��
?�Q��Y��>\)@G�C���                                    Bx��|8  �          @fff?�{�W
=���G�C�.?�{�W�>�ff@���C�#�                                    Bx�׊�  T          @\��?����C�
��=q��33C���?����@��?&ffA/�
C��R                                   Bx�י�  
�          @e�?���AG��.{�.{C��=?���<(�?=p�A?
=C�J=                                   Bx�ר*  "          @j�H?�
=�P  �#�
�$z�C���?�
=�I��?O\)AL��C�
=                                    Bx�׶�  �          @l(�@�
�A녾�=q��ffC�q@�
�>{?&ffA"=qC�`                                     Bx���v  
Z          @hQ�?L���O\)>��@ ��C��?L���C33?�{A���C�Y�                                    Bx���  �          @vff��33�QG�?�=qA�33Cs+���33�333@ ��A�Coff                                    Bx����  
�          @tz��+��G�?�(�A�z�C\=q�+���(�@B  CT�3                                    Bx���h  �          @mp��ff��R?�\A���C_��ff����@ffB�RCV33                                    Bx��               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   Z          @l��?aG��e�aG��UC��=?aG��_\)?aG�A[33C�)                                    Bx��Z  T          @n{?^�R�c�
�W
=�N�RC��R?^�R�]p�?aG�A]��C��                                    Bx��,   �          @j�H?�G��`  ��G��ڏ\C�\?�G��^{?#�
A!��C�!H                                    Bx��:�  T          @tz�?��\�^{�aG��V=qC��?��\�e�>W
=@O\)C��                                     Bx��IL  T          @���?p���xQ�L�Ϳ5C��f?p���mp�?���A��
C�3                                    Bx��W�  "          @�(���
=�z�H?��Ȁ\C��3��
=�J=q@?\)B,�C�8R                                    Bx��f�  T          @���>�{�~�R?�p�A�33C�p�>�{�U�@(Q�BQ�C��                                    Bx��u>  T          @��R����q�?�p�A��C��H����Dz�@333B'�
C��3                                    Bx�؃�  
�          @��>��H����?+�A=qC�t{>��H�i��?��A؏\C��{                                    Bx�ؒ�  �          @�?�p��~{>��@�
C���?�p��n�R?�{A��\C�'�                                    Bx�ء0  T          @�G�?����l�Ϳ5�&{C�N?����o\)>�@�G�C�9�                                    Bx�د�  �          @�ff�
=��33?&ffA�C�녿
=�l��?��A���C�|)                                    Bx�ؾ|  �          @��
?5����>�Q�@��RC���?5�n�R?�=qA��C�`                                    Bx���"  T          @�(�>u���H>�(�@�{C��3>u�p��?�A�33C�ٚ                                   Bx����  T          @���>aG����\?\)@�33C���>aG��l��?�ffA���C���                                    Bx���n  �          @�{?�(��qG���\)�z=qC�K�?�(��j�H?fffAH  C��{                                    Bx���  �          @�  ?�z����\<��
>��C��\?�z��w�?��A�33C�W
                                    Bx���  
�          @��?�Q����=���?�{C�(�?�Q��tz�?�{A��
C��                                     Bx��`  �          @�ff?�Q���Q�>�?�  C�>�?�Q��qG�?�\)A�33C��q                                    Bx��%              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��3�   Z          @���?:�H��{>��R@��C���?:�H�x��?���A��\C�J=                                    Bx��BR  
�          @�  ?&ff��>�\)@p��C�o\?&ff�xQ�?�=qA�z�C��                                    Bx��P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��_�   Z          @���?5����?E�A&�HC��)?5�l(�@�\A��C�s3                                    Bx��nD  �          @��?p�����H?z�@��C�xR?p���l��?�AͅC�                                      Bx��|�  �          @��R?z�H��=q>aG�@?\)C�?z�H�r�\?��RA�
=C�<)                                    Bx�ً�  
�          @�
=?(���p�<�>�C�'�?(��|(�?���A�C�b�                                   Bx�ٚ6  �          @�
=?fff���
=�\)?c�
C�"�?fff�x��?�\)A��C�}q                                   Bx�٨�  T          @�
=?+����?\)@�C��H?+��n{?�A�
=C��                                    Bx�ٷ�  	�          @�  ?xQ����>���@�
=C��=?xQ��p��?�Q�A�  C�@                                     Bx���(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   Z          @��H?W
=��  >�\)@j=qC��)?W
=�|(�?У�A�  C��                                    Bx�� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��f   Z          @�=q?u���?!G�AC�z�?u�o\)?�
=A���C�/\                                    Bx��  
�          @��H?k���{?#�
A33C�33?k��p��?���A�Q�C��                                    Bx��,�  �          @�33?�{���?fffA>�\C��=?�{�fff@�A�Q�C���                                    Bx��;X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��I�   Z          @�p�<����
?0��A�C�7
<��z=q@z�A�33C�=q                                    Bx��X�  
�          @�(�>�  ��=q?&ffAz�C���>�  �xQ�@G�A�C��)                                    Bx��gJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��u�   Z          @�G�?��}p�?Q�A1�C��?��^�R@z�A���C�"�                                    Bx�ڄ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ړ<   Z          @�{?�=q�xQ�?O\)A3�C�|)?�=q�Y��@�\A�C���                                    Bx�ڡ�  "          @��?�33�vff?#�
A
�RC��{?�33�[�?�\)A�33C���                                   Bx�ڰ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ڿ.   Z          @�=q?�ff�p��?���A��C�,�?�ff�Dz�@'
=B�RC�Z�                                    Bx����  �          @�=q?�p��|��?\(�A8��C�C�?�p��\(�@
=A뙚C��                                     Bx���z  T          @���@��j=q?z�@�=qC�Ф@��QG�?�G�A�33C�C�                                    Bx���               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����   Z          @��
@�녿(��<#�
=�C�]q@�녿�R>k�@J�HC���                                    Bx��l  T          @\)@|��<�=L��?8Q�>�(�@|��=#�
=#�
?
=q?&ff                                    Bx��  "          @~�R@w�?&ff>��
@�
=A��@w�?8Q�=�Q�?��\A)                                    Bx��%�  �          @}p�@w
=?&ff��G���(�A��@w
=>�׿!G��33@�z�                                    Bx��4^  �          @~�R@y����Q�.{�
=C�Z�@y�����R�(���HC��R                                    Bx��C  
�          @�z�@Vff��>�{@�
=C�@Vff����?��\Ao33C���                                    Bx��Q�  "          @�(�?k�����?k�AC�C�AH?k��fff@��A�z�C�,�                                    Bx��`P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��n�   Z          @��\�����R?W
=A2ffC�����l(�@p�A��C�C�                                    Bx��}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�یB   Z          @�z�>�\)��?��A�\)C��>�\)�`  @(Q�B�C�Q�                                    Bx�ۚ�  �          @�(��#�
��ff?��HA���C�논#�
�b�\@#�
B��C��                                    Bx�۩�  �          @��?�\��{?�ffA_\)C�xR?�\�e�@=qB�C��                                    Bx�۸4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����   Z          @��
�B�\���
?��HA�C��;B�\�XQ�@1G�BffC�aH                                    Bx��Հ  �          @�z�>�����?�(�A�\)C��q>����`��@$z�B{C�B�                                   Bx���&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����   Z          @���?���w�?�\A��HC�l�?���A�@?\)B0\)C��                                     Bx��r  T          @��>���tz�?�z�A���C�}q>���<(�@G
=B8�C���                                    Bx��  
�          @���>�(��\��@(�B33C���>�(����@^{B[C�R                                   Bx���  
�          @��?333�]p�@p�B��C���?333���@`  BZ33C��                                   Bx��-d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��<
   Z          @��?B�\�l��@z�A�C��?B�\�0��@N{BA��C���                                    Bx��J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��YV   Z          @�=q@�?�=q@P��BT�
A��@�@	��@+�B$��B2{                                    Bx��g�  
�          @��@��#�
@UB\(�C��@�?�G�@L(�BM�RA�{                                    Bx��v�  �          @���@�\�}p�@[�Ba�C�@�\>��@c�
Bo@��
                                    Bx�܅H  "          @�{?�p��+�@	��B�C���?�p���G�@<��BPQ�C��                                    Bx�ܓ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ܢ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ܱ:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ܿ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��Ά  
�          @�(�<��
��  ?��\AW33C�  <��
�g�@��B=qC�%                                    Bx���,  	�          @��?c�
�Z=q@p�B ��C�O\?c�
�=q@QG�BO��C�"�                                    Bx����  �          @�  ?���u�?n{AQC���?���P  @{B��C��                                    Bx���x  
�          @���>��
���H>��@�  C�=q>��
�j�H?�=qA�\)C�}q                                   Bx��	  �          @��H>����Q�    <��
C��
>���~�R?�G�A���C��                                   Bx���  �          @��
�������\����z�C�Uý�����\)?�ffA^�\C�Q�                                    Bx��&j  	�          @�  ?p���}p�?�@�RC��3?p���aG�?��AۅC��H                                    Bx��5  |          @��?�{��  ?8Q�AG�C���?�{�^�R@ffA�
=C��\                                    Bx��C�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��R\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��a            @�G�@9���ff?�A���C�u�@9����Q�@
�HB�C��f                                    Bx��o�  �          @�ff@Q��a�?5A(�C��{@Q��B�\?�
=AۮC��=                                    Bx��~N  
�          @�  @��Y��?n{AK
=C�j=@��5@A�ffC���                                    Bx�݌�  
�          @�Q�@"�\�Vff?(��AffC��{@"�\�8��?���A���C���                                    Bx�ݛ�  �          @�Q�@���Z=q?W
=A6�RC���@���8Q�@G�A��C��q                                    Bx�ݪ@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ݸ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��ǌ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  
�          @�33?�  �|(�?�@��
C��?�  �^�R?�A��C�O\                                    Bx���~  
(          @�=q@(��l��>k�@B�\C�O\@(��W�?���A�Q�C���                                    Bx��$  
�          @�33@z��j�H=�G�?�p�C�&f@z��W�?���A���C�AH                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��.  
          @��H@
=q�p�׾���   C��3@
=q�c33?��RA���C���                                    Bx��<�  T          @�=q@{�`�׾�p���C��{@{�Y��?s33AK\)C��                                    Bx��Kb  
�          @�z�@ff�u��G���33C�W
@ff�g
=?���A�{C��                                    Bx��Z  
P          @�p�@�\�x�þǮ����C��\@�\�p��?�=qAbffC�9�                                    Bx��h�  
�          @�(�?��w
=�xQ��L��C��\?��|(�?�@�
=C�S3                                    Bx��wT  
�          @��
?�(��r�\�h���AG�C��)?�(��w
=?\)@�=qC��f                                    Bx�ޅ�  �          @���@33�r�\�\(��4Q�C�5�@33�u?(�@��C�\                                    Bx�ޔ�  
�          @�p�?��w��\(��4z�C�.?��z�H?#�
AC�\                                    Bx�ޣF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�ޱ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  	�          @��
@(��c33�!G���C�E@(��a�?=p�AG�C�Z�                                    Bx���8  T          @�(�@ff�g
=�B�\�\)C���@ff�hQ�?&ffA	�C�t{                                    Bx����  
�          @��
@  �j�H�333�(�C��R@  �j�H?:�HAC��q                                    Bx���  
�          @��@(��`  �����G�C�o\@(��Y��?n{AI�C���                                    Bx���*  
�          @���@5�J�H>\)?�{C��@5�8Q�?���A��RC�J=                                    Bx��	�  
Z          @��@B�\�A�>8Q�@�HC���@B�\�/\)?��A�=qC�                                      Bx��v  
�          @���@C33�?\)=u?W
=C�Ǯ@C33�/\)?�Q�A�G�C�H                                    Bx��'  �          @���@:�H�Dzᾞ�R����C���@:�H�=p�?aG�A>�RC�U�                                    Bx��5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��Dh              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��S  J          @��\@1G��K��u�L(�C���@1G��S�
>���@��HC��)                                    Bx��a�  �          @�=q@,(��I����p���=qC�>�@,(��XQ�=u?\(�C�AH                                    Bx��pZ  "          @��?�=q�|�Ϳ�=q�i��C�P�?�=q���?   @׮C��                                    Bx��   !          @��?ٙ��i�����
����C�xR?ٙ��w
=>W
=@5C���                                    Bx�ߍ�  �          @�G�?�  �n{������C��{?�  �y��>��
@��C�{                                    Bx�ߜL  "          @�  ?�=q�dzῧ���(�C��3?�=q�s33>.{@\)C��q                                    Bx�ߪ�  �          @���?��R�^{���\�ep�C��=?��R�fff>Ǯ@�p�C�z�                                   Bx�߹�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  
�          @��
@(��7
=����
=C�@ @(��Q녾�(���  C�T{                                    Bx���  
�          @�=q@{�p��
=���C��=@{�E��}p��a�C�Q�                                    Bx���0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��|  
�          @���@z��   �2�\�0ffC��{@z��;������=qC���                                    Bx�� "  �          @���?�p�����,���)�\C�޸?�p��P�׿�ff��Q�C��3                                    Bx��.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��=n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��L  
�          @�z�>���C�
�.{�%��C�q�>���x�ÿ�ff��  C��                                    Bx��Z�  �          @��?u�C�
�#�
�G�C��\?u�tzΐ33��ffC�
=                                    Bx��i`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  I          @�ff@%������?\)�6=qC��\@%����������C�G�                                    Bx���R  T          @��@,(���\)�=p��5��C��{@,(��������
C�e                                    Bx���  "          @���@#�
�aG��Fff�C��C���@#�
���   �C��f                                    Bx�ಞ  T          @�z�@0  �k��;��5
=C��@0  �   ���	ffC�                                      Bx���D  
�          @�
=@�Ϳ���O\)�I�C�\)@���  �#33��HC��
                                    Bx����  
�          @��@������Tz��V\)C�l�@������:�H�4�C�Q�                                    Bx��ސ  �          @�z�@\)����J�H�N��C�@ @\)��33�5�3ffC�U�                                    Bx���6  "          @�ff@*=q���\�?\)�9ffC��f@*=q�Q����	��C���                                    Bx����  �          @�ff@333�5�A��9\)C�� @333��{�\)�(�C�1�                                    Bx��
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��'�  I          @�G�@@  ��p��*�H�%�C�~�@@  ��z��33�  C�h�                                    Bx��6t  "          @��@J=q����� ���
=C�J=@J=q���
�
�H���C��R                                   Bx��E              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��S�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��bf  	�          @���@I���u�   �C�|)@I���������	{C��\                                    Bx��q  T          @\)@G
=�����{�C�P�@G
=��\)�ff���C�%                                    Bx���  
�          @�Q�@J=q���(��z�C��q@J=q��G��{��C�q                                    Bx��X  
�          @�G�@XQ�>�\)�	���p�@���@XQ��R����
=C���                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�᫤              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��J  
          @�G�@_\)�L��� ����C�^�@_\)��  ��G��θRC���                                    Bx����  
�          @��H@Z=q�8Q��
=��
=C��@Z=q��G��У�����C��                                    Bx��ז              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  I          @�=q@XQ�\����p�C�Ǯ@XQ쿠  �������C���                                    Bx���  T          @���@[���������C�q@[����
���H��=qC��q                                    Bx��.  �          @�G�@`�׾��H��
=��\C��@`�׿��R�Ǯ���C�B�                                    Bx�� �  �          @���@dz���H������{C�\@dzῙ���������RC��
                                    Bx��/z  "          @��@g
=��������33C�:�@g
=�������R����C���                                    Bx��>   "          @�@tz᾽p��У���p�C�8R@tz῁G���=q��=qC��R                                    Bx��L�  T          @��H@i���Q녿�����{C���@i����33����yG�C���                                    Bx��[l  
�          @\)@e�aG������HC�H�@e�c�
��
=���HC�\                                    Bx��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��x�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��^  I          @��@~{���Y���?
=C���@~{�E��z����C�~�                                    Bx��  "          @�33@k���׿У�����C�^�@k���{���
���
C���                                    Bx�⤪  �          @~�R@\��>�
=�������@�  @\�;�\)�У����C��                                    Bx��P  �          @��@|�ͿTz�.{���C��@|�Ϳ����=q�qG�C���                                    Bx����  �          @���@vff��(��#�
�z�C�/\@vff��
=>�Q�@��C�|)                                    Bx��М  �          @��@s33����<#�
>B�\C���@s33��  ?
=A
=C��H                                    Bx���B  �          @��H@x�ÿ��׾#�
��C��@x�ÿ��>���@���C�'�                                    Bx����  T          @��H@s33�@  ��p���ffC�g�@s33�����J=q�4z�C�@                                     Bx����  
�          @��
@Z=q�����p���C���@Z=q��33���
�l��C�s3                                    Bx��4  T          @���@>�R�-p���Q쿚�HC��q@>�R�\)?���Au��C��                                    Bx���  
�          @�(�@`  ��þB�\�,(�C�L�@`  �G�?:�HA#�C�f                                    Bx��(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��7&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��E�  
�          @�z�@C33�1G��.{�C�޸@C33�%�?��\Af{C��                                    Bx��Tr  T          @�{@ff�L(���p����
C�0�@ff�Z=q>u@W�C�J=                                    Bx��c  
�          @�
=@(���(�<��
>��C�� @(�����?�ffA�Q�C�
                                    Bx��q�  T          @�@QG���?��A�  C�/\@QG�����@(�A�
=C���                                    Bx��d  �          @�ff@Q���
?�
=A��C�c�@Q녿˅@�
A�C�3                                    Bx��
  
�          @�
=@E��   ?�
=A�
=C�u�@E���33@�Bz�C��                                    Bx�㝰              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��ɢ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���H            @��@e��R=�Q�?�(�C��@e��p�?��Ad��C��3                                    Bx����  
�          @��@_\)�  ?O\)A1�C���@_\)��
=?ٙ�A��C�'�                                    Bx����  �          @�  @`  �Q�?k�AK�C�O\@`  �\?�G�AŮC�8R                                    Bx��:  �          @���@R�\�,�ͽ��
���\C�O\@R�\�{?���Ao�C���                                    Bx���  �          @��@Fff�<�;���33C�+�@Fff�7
=?^�RA:{C��H                                    Bx��!�  
�          @��@@���*�H���R����C�1�@@���C33�8Q��p�C�H�                                    Bx��0,  I          @��@@���2�\��p���G�C���@@���C33=�G�?�C�S3                                    Bx��>�  "          @��@A��<�Ϳ8Q���HC��@A��>{?!G�A�C�˅                                    Bx��Mx  �          @��@L���,(��}p��T��C��R@L���6ff>�\)@qG�C�%                                    Bx��\  �          @��@P���\)������C�XR@P���3�
��Q쿞�RC��                                    Bx��j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��yj              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��  
O          @�
=@>�R�=q��Q���
=C�}q@>�R�333�u�W
=C�`                                     Bx�䖶  "          @��@Q녿333?��A��C��@Q�>#�
?�z�A�@1�                                    Bx��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��¨  
�          @�ff@xQ�^�R?��A�=qC���@xQ�W
=?�ffA���C�k�                                    Bx���N  �          @�{@�Q��?�=qAnffC���@�Q��?�(�A�
=C�                                    Bx����  "          @�p�@�G��
=q?Y��A<(�C�33@�G���G�?�  A^=qC�9�                                    Bx���  �          @�ff@�33��G�?@  A$Q�C��@�33��\)?^�RA>�RC���                                    Bx���@  
�          @�{@�33�0��>��H@�{C�0�@�33���?=p�A$  C�(�                                    Bx���  �          @��@�Q�p��?\)@���C�` @�Q���?k�AL��C���                                    Bx���  �          @�p�@~�R��z�>�z�@���C���@~�R�h��?G�A,z�C��3                                    Bx��)2  �          @�z�@z=q��G���G���=qC�
@z=q��
=>�@�(�C��                                     Bx��7�  
(          @�(�@i����Q�s33�S�C���@i����
=���Ϳ��C��                                    Bx��F~  "          @��@c33����h���O
=C���@c33� �׼��
��\)C�9�                                    Bx��U$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��rp  I          @�(�@r�\��ff�����HC��@r�\�˅>�\)@}p�C��                                    Bx��  "          @��@hQ��׿@  �&�\C�H�@hQ��G�>.{@�C�q�                                    Bx�叼  T          @�p�@r�\��녿z�� ��C�L�@r�\��p�>W
=@<(�C��                                     Bx��b  T          @��@b�\��G������x(�C��@b�\��
�B�\�)��C��)                                    Bx��  �          @�\)@G
=�'�����n�HC��3@G
=�5�>W
=@5C��q                                    Bx�廮              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx����  
Z          @��
@p  �L�Ϳ����C�p�@p  �p�׿�33��G�C��q                                    Bx���  
�          @�Q�@L(��2�\�����C�aH@L(��.�R?J=qA)C���                                    Bx���F  "          @���@\�����Ǯ����C�l�@\���?@  A#
=C��                                    Bx���  -          @�\)@vff��p��W
=�7�
C��H@vff�ٙ����
����C�)                                    Bx���  �          @�(�@x�ÿY������u��C�޸@x�ÿ�  �
=��C��                                    Bx��"8  "          @�(�@x�ÿ(���  ����C���@x�ÿ�{�Tz��:ffC��                                    Bx��0�  
�          @�33@x�þ�\)��(����RC���@x�ÿO\)�u�X��C�'�                                    Bx��?�  
�          @��@w��.{��{��z�C���@w��E�������(�C�ff                                    Bx��N*  "          @��\@tz�>����z���ff@z�@tz��\��=q��=qC�9�                                    Bx��\�  "          @��\@p  >��ÿ����z�@��@p  ��녿\��=qC���                                    Bx��kv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���            @��@E��2�\��ff����C���@E��E�=�?��C��f                                    Bx��h  �          @���@C33�<�Ϳ�\)�lz�C��q@C33�HQ�>�Q�@��C��                                    Bx��  "          @�(�@?\)�@  ����ap�C�w
@?\)�J=q>�(�@�p�C��                                     Bx�洴  �          @��H@5��J�H�^�R�8  C��H@5��Mp�?0��A{C���                                    Bx���Z  T          @��H@0  �O\)�Y���5��C�  @0  �QG�?:�HA(�C�                                      Bx���   
�          @��\@-p��R�\�B�\� ��C���@-p��QG�?W
=A3�
C���                                    Bx���  �          @�(�@�R�dz����33C�aH@�R�Y��?�(�A�G�C�3                                    Bx���L  �          @�(�@-p��W��#�
�ffC�` @-p��Q�?}p�AQp�C��q                                    Bx����  "          @��@���c33�O\)�,Q�C��q@���aG�?k�AB�\C��                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��)�            @�=q@�R�Z�H�����  C���@�R�hQ�>�ff@�\)C��f                                    Bx��8�  T          @�33@Q��Y������{C�
=@Q��j�H>��
@���C��                                    Bx��G0  �          @���@:�H�1G���  ��p�C�Ff@:�H�I���#�
���C�o\                                    Bx��U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��d|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��s"  
�          @�G�@:=q�\)��z���ffC���@:=q�Fff����33C��3                                    Bx���  
�          @��H@!��P  ���
��Q�C��@!��^{>\@��C�                                    Bx��n  
�          @��
@   �U��p���G�C�h�@   �a�>��@�Q�C��=                                    Bx��  
�          @���?��H�c�
����̏\C�Ф?��H����    �L��C�|)                                    Bx�筺  
�          @�z�?�{�U��
=����C���?�{�{������{�C��\                                    Bx��`  "          @��?��\�Y���=q�z�C�B�?��\��(��
=q���HC��                                    Bx���  "          @�{?�33�b�\�p����HC�˅?�33�����z��p  C�L�                                    Bx��٬  
�          @��?����R�\�
�H��RC�H�?����{���p����\C�,�                                    Bx���R  
Z          @��?�p��G�����p�C�:�?�p��tz�����
=C���                                    Bx����  T          @��@{�QG���=q��p�C��@{�`��>�p�@��C��
                                    Bx���  T          @���@A��C33?Tz�A.{C�` @A��33@
�HA�C�]q                                    Bx��D  T          @��@<���O\)>�
=@��C�'�@<���*=q?�33A�  C��R                                    Bx��"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��@6  
�          @���@1��S33����  C�
=@1��K�?��Ab{C���                                    Bx��N�  "          @��@  �p  �   ��G�C�w
@  �b�\?��A�\)C�4{                                    Bx��]�  
�          @��R@{�s�
�5�33C��@{�l(�?�
=At��C�~�                                    Bx��l(  T          @�ff?�\��G������=qC��
?�\�p  ?ǮA��\C���                                    Bx��z�  
�          @�?�z��u�}p��O�
C�<)?�z��vff?uAHQ�C�5�                                    Bx��t  �          @�@�
�w��
=��z�C�H@�
�k�?���A�\)C���                                    Bx��  T          @�p�@#33�\�Ϳfff�=G�C�5�@#33�^{?Y��A3�
C�*=                                    Bx���  T          @��R?�(�����>�z�@p��C��q?�(��g
=@�\A��C�XR                                    Bx��f  T          @�{?��R���=u?J=qC�"�?��R�l��@�A�ffC�AH                                    Bx���  
�          @�{?�=q����=#�
?��C�f?�=q�p��@A�33C��q                                    Bx��Ҳ  
�          @�ff?�\)���׾�{��p�C�U�?�\)�z�H?޸RA�\)C��R                                    Bx���X            @�{?�����{���
����C�^�?����u?�(�A���C�                                    Bx����  T          @���?�  ��G�?(��A
�HC�0�?�  �N�R@ ��B�
C�u�                                    Bx����  
�          @�z�?�  ��G�?�@�=qC�.?�  �R�\@=qB��C�@                                     Bx��J  T          @�z�?�33��ff��ff���\C���?�33�y��?У�A�z�C�8R                                    Bx���  
�          @��?����{����  C��
?���|��?\A��HC�`                                     Bx��*�  "          @��?�����ff��33���C�XR?����vff?��HA�G�C���                                    Bx��9<  �          @���?(����
=>L��@,��C�xR?(���fff@p�A��RC�9�                                    Bx��G�  "          @�G�?fff���u�@  C�{?fff�l(�?�(�A���C���                                    Bx��V�  T          @�  ?=p�����u�Y��C�3?=p��j�H?���A�{C���                                    Bx��e.  �          @�G�?E��xQ�?��RA���C��
?E��333@=p�B6
=C���                                    Bx��s�  "          @��
?�  �mp�?�33A���C��?�  ���@O\)BA��C��f                                    Bx��z  "          @�Q�?�G��h��@��A�33C���?�G��@o\)Bc\)C��\                                    Bx��   
�          @�\)?����|(�?�  A�ffC���?����'
=@\(�BI��C��3                                    Bx���  
O          @�
=?B�\�^{@)��B��C�7
?B�\��  @\)B��C��=                                    Bx��l  �          @���>�33��?�{A��
C�` >�33�7�@\(�BGz�C�s3                                    Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��˸              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���^  I          @�G�?�R��(�?��
A�z�C�H�?�R�/\)@c�
BN�
C�e                                    Bx���  "          @��R?�=q����?��A��\C���?�=q�0��@S�
B@\)C���                                    Bx����  "          @��@AG��8��?�@�z�C�%@AG��G�?�{A�=qC��\                                    Bx��P  "          @��
@j�H�
�H��ff��  C��
@j�H�
=?+�A33C��                                    Bx���  �          @���@~�R���ÿ^�R�;�C��\@~�R��=q��G����
C�.                                    Bx��#�  �          @�\)@vff��=q�xQ��T��C�t{@vff��녾B�\�&ffC��H                                    Bx��2B  �          @��@w�>k���\��{@W
=@w��+���33����C�q                                    Bx��@�  �          @�  @n�R?&ff�����HA33@n�R��p����R��  C�/\                                    Bx��O�  T          @�  @|(�>�׿�p�����@ڏ\@|(����R�\��Q�C���                                    Bx��^4  �          @���@Y��?�=q��33����A��@Y��?h���
=��  Ap��                                    Bx��l�  "          @�
=@i��?��R��33��{A��
@i��?(�������=qA��                                    Bx��{�  
�          @��R@l��?��\��p����A�@l��>\����@�(�                                    Bx��&  "          @�
=@p  ?��H��p�����A���@p  >��
�������@��                                    Bx���  "          @�
=@�G�?B�\�c�
�C\)A+�@�G�>k���33�~{@N�R                                    Bx��r  �          @�\)@��
>L�Ϳ333�p�@5@��
�#�
�5��C���                                    Bx��  �          @���@�ff=#�
�G��(��?
=q@�ff��33�333��RC��q                                    Bx��ľ  "          @���@�Q�>�=q���
����@dz�@�Q�=�\)������?}p�                                    Bx���d  "          @���@��>�{�\)��G�@���@��<#�
�&ff���>B�\                                    Bx���
  �          @��@�ff?녿=p��p�@��@�ff>��k��E�?�                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���            @��H@�Q�u?�RA��C�j=@�Q�=���?(��Az�?��                                    Bx���  �          @��@���W
=?�  AT��C��R@��>��
?xQ�ANff@���                                   Bx��+H  
�          @�33@��׼��
?=p�A=qC�� @���>�33?(��A33@�\)                                   Bx��9�  "          @�
=@�Q쿂�\?��RA��RC��@�Q�\)?�ffA�G�C�                                     Bx��H�  �          @�p�@�33�u>�ff@�{C�ff@�33<#�
?�\@׮>B�\                                   Bx��W:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��t�  
�          @��@��L��?&ffAp�C��@�>�\)?
=@�\@dz�                                    Bx��,  T          @�@w�?�@"�\BA{@w�?�G�?�z�A��
A�p�                                    Bx���  `          @�z�@q�?�Q�@G�A�=qA��R@q�@Q�?�  AyB �R                                    Bx��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���  H          @���@r�\?�
=����Q�A��
@r�\?�
=������
=A��                                    Bx���j  T          @�=q@h��?z�H�u�c�
Ap��@h��>�p�������z�@���                                    Bx���  T          @�33@xQ�>�z�����Q�@�\)@xQ�0�׿�\���C��\                                    Bx���  �          @�33@���?녿������HA Q�@��׾�z�����33C��f                                    Bx���\  �          @�{@u?@  ��\)���\A/�
@u�#�
�Ǯ����C��                                    Bx��  �          @\��@(�ÿ�p���ff��  C�}q@(�ÿ�(��   �G�C�z�                                    Bx���  �          @��H>����R�n{�G33C�� >����?��A�p�C��                                    Bx��$N  �          @��\���
���\>��?�
=C|}q���
��33@#�
A�ffCy�)                                    Bx��2�  T          @����X���AG�?xQ�A@��C]�3�X����@
=A�  CT\                                    Bx��A�  �          @�
=�XQ��aG�?fffA&�HCb8R�XQ��%@#33A�G�CY�                                    Bx��P@  �          @�
=�@  �u�?L��A��Cg޸�@  �9��@'�B 33C`
=                                    Bx��^�  "          @�����\��p�>�Q�@�p�Cr�H��\�h��@"�\A��HCm�\                                    Bx��m�  �          @���(����R>�33@��Cs�f�(��j�H@#33A���Co@                                     Bx��|2  �          @���(����(�>�
=@��
Cmu��(���W
=@��A�z�Cgٚ                                    Bx���  �          @�(��L(��l(�>��
@q�Ce+��L(��AG�@��A��C_xR                                    Bx��~  T          @�z��B�\�r�\=���?��HCgE�B�\�N{@   A��HCb��                                    Bx��$  �          @�z��+����H�8Q��Cl�\�+��g�?�A�ffCi�                                    Bx���  "          @�ff�{���
�
=���Cs��{��=q?��HA��Cqh�                                    Bx���p  �          @�z����33�G��\)CtQ������?��
A��Cs:�                                    Bx���  T          @��׿����33�#�
�.{Cv�׿���p��@(�A�z�Csn                                    Bx���  T          @�  �����p��W
=��RCq�\����l��?�Q�A��RCn�
                                    Bx���b  �          @��H�(���녿u�6�\Co��(��~�R?��RAmp�Cn��                                    Bx��   �          @�p��!��vff��\���Cl���!���
=>�(�@���Co
=                                    Bx���  T          @��H����n{��\)���\Cl�{�������>�=q@P��Cop�                                   Bx��T  �          @�Q�����g�����{Cq� �������ý����RCu�=                                   Bx��+�  �          @�녾\)�|(��/\)�
=C��
�\)��G���33��ffC�%                                    Bx��:�  �          @�G�>�=q�g
=�G
=�"�RC�q>�=q��{�W
=�"ffC��H                                    Bx��IF  �          @��׾����
������C�q����Q�=#�
>��C�<)                                   Bx��W�  "          @�=q�!G��x���0����
C�ff�!G���Q�Ǯ��(�C�9�                                   Bx��f�  �          @�  ��\)�XQ��S�
�1Q�C��q��\)���H�����[�
C�>�                                    Bx��u8  �          @��H�!G��&ff�l(��XffC�+��!G����
��������C���                                    Bx���  �          @�G����N{�A��+�C��f����=q�u�F�\C��                                    Bx�풄  �          @�Q�>��
���Ϳ@  �#
=C�+�>��
�|��?�p�A���C�G�                                    Bx���*  �          @�Q�?�����?�(�A{33C�l�?��Dz�@O\)B8��C��f                                    Bx����  �          @�p�?!G���z�?��A�  C�Q�?!G��7�@QG�B@(�C�.                                    Bx���v  �          @�=q?\)���?��AW�
C���?\)�Mp�@H��B/��C���                                    Bx���  �          @��H?8Q����H?fffA8��C���?8Q��QG�@@  B'Q�C�0�                                   Bx����  �          @�(�?=p�����>.{@�C��)?=p��tz�@{B��C�xR                                   Bx���h  �          @���?k���{�u�E�C��q?k��u@{A���C��q                                    Bx���  �          @��\?����|(�������C�=q?������\>�G�@�  C�u�                                    Bx���  "          @��H?�33�Z�H�%���RC��?�33��  ����\)C��R                                    Bx��Z  �          @�{?�33�p�׿�
=�ͮC�5�?�33��
=>�=q@aG�C�.                                    Bx��%   T          @�p����R����@1G�B	�HC��쾞�R��@�{B�k�C��{                                    Bx��3�  
�          @�(������=q@
=qA�
=C�⏾���&ff@�=qBd�C���                                    Bx��BL  "          @��\������H@�
A�z�C�������*=q@�  B_�C���                                    Bx��P�  T          @��R������  ?��\A|��C�Z�����L��@Z=qB;{C�R                                    Bx��_�  �          @�{>8Q����H>�@�\)C�)>8Q��l��@0��B��C�`                                     Bx��n>  �          @�=q>�z����׾W
=�,(�C�Ф>�z��~�R@��A�  C�\                                    Bx��|�  �          @��>�p���33���أ�C�h�>�p��\)?�ffA��C���                                    Bx�  "          @��\?@  ���>�{@��
C�
?@  �X��@(�BffC�7
                                    Bx��0  �          @��
�O\)��{?Q�A.�\C���O\)�J�H@7�B%{C��                                    Bx���  �          @�Q�@  �xQ�?(�A��C����@  �@��@!�BC�                                    Bx��|  T          @�G�?��k�?��A��\C�` ?��#�
@8Q�B>�C�<)                                    Bx���"  
�          @hQ�   �Fff?�Q�A�z�C�h��   ���
@C33Bk{C~J=                                    Bx����  �          @U�?Q���H�����
=C�S3?Q���?h��A�Q�C��)                                    Bx���n  "          @L��?�G�?
=q�!��W�A��?�G��Y���(��M33C�/\                                    Bx���  T          @h��?�p��\�9���\��C���?�p���=q���!{C���                                    Bx�� �  �          @��
@p��h���N�R�U��C��{@p������\�(�C��                                    Bx��`  �          @��\@&ff��G��R�\�G{C�ff@&ff�$z��33� ��C��=                                    Bx��  T          @�Q�@J=q��33�L���5  C��\@J=q���H�#�
�
�C��                                    Bx��,�  
�          @�G�@H�ÿxQ��HQ��.Q�C�h�@H������
�H��RC��                                    Bx��;R  �          @��\@c�
���
�,(����C��@c�
�녿�G���G�C���                                    Bx��I�  �          @��@Z=q����*=q��C�Ф@Z=q�,�Ϳ�(���C��                                    Bx��X�  "          @�@e����
�3�
�ffC��{@e����\)��
=C�b�                                    Bx��gD  �          @�\)@^�R�u�C�
�!{C�L�@^�R�=q�
=��z�C��=                                    Bx��u�  �          @�ff@\(������@  �
=C�T{@\(��{�   ��z�C�!H                                    Bx�  T          @���@Z=q��ff�7����C�7
@Z=q����{�홚C���                                    Bx��6  �          @�(�@o\)���
�!G����C�L�@o\)�(���\)����C��3                                   Bx���  �          @��@p  ��
=�!G���RC�J=@p  ��
�����(�C�%                                   Bx�ﰂ  
�          @�@u����R����C�  @u��33��������C���                                    Bx��(  T          @���@xQ�k��Q����HC�U�@xQ��G���ff��z�C�8R                                    Bx����  
�          @�
=@�p��h�ÿ��R��p�C��R@�p�����(��p(�C�E                                    Bx���t  H          @�\)@�녿+�����Q�C��{@�녿�G���Q��j�RC�O\                                    Bx���  �          @�{@��H���ٙ���ffC��R@��H��ff�����m�C��                                    Bx����  �          @�Q�@��׿����\)��=qC�9�@��׿�p��\(��&�\C�e                                    Bx��f  �          @���@��\�\)��G��t��C���@��\���׿J=q�\)C��                                    Bx��  �          @�=q@���J=q��z��_�C��@�����
�z�����C�7
                                    Bx��%�  T          @��\@�����
=��=q�O
=C��q@������þ��R�hQ�C�|)                                    Bx��4X  T          @�33@�����Q�O\)�C�=q@�����33=u?8Q�C��                                    Bx��B�  �          @�(�@��ÿ��ÿh���+\)C�q�@��ÿ�=L��?��C��                                    Bx��Q�  �          @��H@��H��=q����Pz�C���@��H�Q�<�>�{C��=                                    Bx��`J  "          @���@��������G�����C�E@����;���I��C���                                    Bx��n�  
�          @�Q�@�G�����p��q�C�(�@�G����=u?333C�aH                                    Bx��}�  
�          @�
=@��׿�\)������ffC�~�@�����H��Q�����C�w
                                    Bx���<  
�          @�\)@�=q��\��(��pQ�C���@�=q��<�>��RC�޸                                    Bx���  �          @��@���G��
=��=qC��@���  ?0��A��C���                                    Bx��  `          @���@�33�ff�333�=qC�@�33��?#�
@�{C��R                                    Bx��.  H          @�\)@z=q�p��s33�9G�C��@z=q�%>�@��
C�=q                                    Bx����  "          @�{@l(��*�H��=q�UG�C�3@l(��5>�@�(�C�:�                                    Bx���z  
�          @�G�@k�� �׿\(��.�HC���@k��%?z�@�(�C�p�                                    Bx���   
�          @���@p  ��R������33C�H�@p  �z�?uAC�
C�"�                                    Bx����  T          @�  @u����33���C�L�@u��
�H?n{A?33C�8R                                    Bx��l  �          @�  @j�H�!녿\)���C��3@j�H���?Y��A.�HC�#�                                    Bx��  T          @�  @b�\�*=q�8Q��z�C���@b�\�(��?G�A z�C��q                                    Bx���  `          @�p�@g
=� �׽��
��G�C��{@g
=��?�  A�ffC�q�                                    Bx��-^  
�          @�(�@a��$z�>#�
@�C��
@a��ff?�p�A���C���                                    Bx��<  T          @��@w
=�=q�B�\�ffC�f@w
=�	��?�{A`��C�u�                                    Bx��J�  �          @���@j=q�&ff>���@u�C�J=@j=q��
?�\)A��C�U�                                    Bx��YP  "          @��R@G
=�>�R?O\)A+33C�)@G
=�ff@��A�=qC��                                    Bx��g�  "          @��
@=p��:�H?}p�AW\)C���@=p����H@Q�BQ�C�G�                                    Bx��v�  "          @�z�@(��I����\��z�C�l�@(��p  ���
���C�%                                    Bx��B  �          @��H@G
=�1녾\)���C�\@G
=�(�?���A���C��                                    Bx���  
�          @���@C�
�E��R��HC�c�@C�
�<��?��Af�\C�                                    Bx��  
�          @��
@H���9���^�R�8(�C���@H���:�H?J=qA&=qC���                                    Bx��4  
Z          @��
@Fff�:�H�u�K
=C�T{@Fff�?\)?8Q�A��C�H                                    Bx���  
�          @�Q�@>�R�Mp��^�R�3�C�g�@>�R�L(�?s33AE�C���                                    Bx��΀  
(          @�{@I���S�
��G��G�C�˅@I���Vff?aG�A.{C���                                    Bx���&  "          @���@@���K���=q�]p�C��3@@���QG�?B�\A
=C�K�                                    Bx����  
�          @�(�?��
�*�H?޸RA뙚C��?��
����@8��BZ�C�G�                                    Bx���r  T          @�ff?�  ����>B�\@p�C���?�  �c�
@�BQ�C���                                    Bx��	  
Z          @��
>����G���\)�k�C�  >���r�\@�A���C�!H                                    Bx���  �          @�(�@�
�Z�H�^�R�Ep�C��@�
�W�?��Ap��C���                                    Bx��&d  T          @��@0���I�����z{C��
@0���S33?+�A�C���                                    Bx��5
  �          @���@.�R�2�\�aG��K
=C�1�@.�R�5�?8Q�A'
=C��R                                    Bx��C�  
�          @�=q@N�R����Q쿡G�C��H@N�R�
=?��HA�
=C�ff                                    Bx��RV  
�          @�z�@xQ��p������(�C�}q@xQ��
=?&ffA�
C��=                                    Bx��`�  T          @��\@HQ���׿aG��L��C�)@HQ��Q�>�ff@�\)C�b�                                    Bx��o�  �          @�ff?�p��'
=@C�
B1��C��?�p����@~�RB��\C���                                    Bx��~H  �          @�=q?�
=�"�\@P��B6=qC��)?�
=��G�@��B�� C��{                                    Bx���  "          @��
@��+�@Dz�B&=qC���@��(��@���BqC���                                    Bx��  T          @��H@�
�7
=@/\)B�RC���@�
�z�H@u�Bc
=C��=                                    Bx��:  T          @��@���5@%�B
  C�` @�Ϳ�ff@l(�BX�C�c�                                    Bx���  
�          @��
@'��>{@A��RC���@'����
@c33BJz�C��
                                    Bx��ǆ  �          @�p�@1��<(�@A��C��R@1녿�  @b�\BE  C���                                    Bx���,  
�          @��@;���@%�Bz�C�.@;��333@^{BD33C�L�                                    Bx����  T          @�ff@ ���7
=@1�B��C���@ �׿u@w�B]
=C���                                    Bx���x  �          @�
=@!��:=q@.{B��C�z�@!녿��
@vffBZ�\C��H                                    Bx��  �          @�
=@#�
�P  @�RA�z�C�)@#�
��=q@g
=BH��C�'�                                    Bx���  
�          @���@#�
�Vff@
=A�{C���@#�
��p�@c�
BD{C��
                                    Bx��j  �          @���@,(��P��@	��A׮C���@,(��У�@c33BA�RC�k�                                    Bx��.  T          @�=q@ ���Z�H@�A�  C�'�@ �׿�Q�@o\)BL
=C���                                    Bx��<�  "          @�z�@\)�i��@z�A�C���@\)��\)@y��BT�
C�3                                    Bx��K\  
4          @�z�@'
=�p��?�z�A�33C�aH@'
=�z�@Y��B0�C�7
                                    Bx��Z  
�          @�z�@��tz�@   A��C��H@��(�@n{BI  C�}q                                    Bx��h�  "          @��R?����\��\)�c�
C�\)?��mp�@�RA���C��                                    Bx��wN  
�          @���?������\>\)?�(�C���?����X��@�A�p�C��R                                    Bx���  �          @�=q@�R�o\)?8Q�A�C���@�R�333@%B�C�                                    Bx��  
�          @�  @���n�R?(�@��HC��\@���5@\)Bz�C�o\                                    Bx��@  
�          @�33@!��W�=�\)?z�HC�|)@!��4z�?�AυC���                                    Bx���  
�          @z�H?�p���
=�p��:C�{?�p��&ff�}p���G�C��=                                    Bx����  �          @�33�G�>������Q�C���G���Q��l(��q�Cx8R                                    Bx���2  �          @��?(��n{�$z��	=qC���?(����׾W
=�*=qC��{                                    Bx����  �          @���?0���W
=�^�R�5��C���?0����ff���]�C�.                                    Bx���~  
�          @��H>��l(��Fff�
=C��q>����׿+����C��f                                    Bx���$  �          @�\)>8Q��z=q�����C�T{>8Q����>��@�\)C�4{                                    Bx��	�  
�          @�33<��}p���Q���
=C�5�<���Q�?(��A��C�1�                                    Bx��p  T          @�=q>��R�c33�(��	�\C��>��R��녾B�\� ��C�{                                    Bx��'  �          @�  ?��������Q��
C���?���a��G
=�$  C��\                                    Bx��5�  �          @��?0�׿����R\C�n?0���x���4z��{C��                                    Bx��Db  �          @��\?O\)�������C��q?O\)�a��J�H�%
=C�w
                                    Bx��S  T          @��?Y���������
=C��3?Y���Tz��P���.=qC�*=                                    Bx��a�  �          @���?��������C�O\?��e�E��!�C�8R                                    Bx��pT  �          @��
>W
=�33�����vQ�C���>W
=��
=�����33C�j=                                    Bx��~�  `          @�33?녿����(��qC�ff?��z�H�.{�
{C�                                      Bx��  H          @�?Q녿W
=��Q���C�)?Q��Fff�XQ��9�C�e                                    Bx���F  "          @�  ?J=q�����G��C�U�?J=q�XQ��O\)�,ffC��R                                    Bx����  	�          @��R?u��z���(�k�C��?u�aG��>{��\C���                                    Bx����  
�          @��?����H����v�HC�b�?��y���#�
� \)C�O\                                    Bx���8  �          @��H?�{��z������{(�C�3?�{�y���(Q��
=C��3                                    Bx����  
�          @�  ?�ff�1G��L(��8�C��
?�ff��G���p����C��                                    Bx���  T          @�Q�?�p��tz��
=�ͅC���?�p���Q�>Ǯ@�
=C��                                    Bx���*  �          @��?�  �q�����
=C�/\?�  ���>k�@>�RC��                                    Bx���  
�          @�=q?@  ��p�����{C�!H?@  ��p�?J=qA (�C��R                                    Bx��v  �          @��?�p��:�H�Dz��,�HC�t{?�p���33����[
=C���                                    Bx��   �          @�p�?����
=�}p��~�C���?���S�
�&ff��C��                                    Bx��.�  �          @�{?�{���R�w
=�q�\C�� ?�{�S�
�\)�33C��)                                    Bx��=h  
Y          @���?��Ϳ��\�|���qC��?����=p��3�
��RC�5�                                    Bx��L  �          @�\)?�=q��33���qC���?�=q�Tz��I���'��C��)                                    Bx��Z�  �          @��
?��R������\)�r�\C�o\?��R�s�
�'
=���C��3                                    Bx��iZ  �          @��H?�Q쿼(���(���C�*=?�Q��c�
�;��ffC���                                    Bx��x   
�          @��\?�G��z�H���R  C���?�G��L(��P���+
=C���                                    Bx����  �          @��\?�
=���
��.C�
?�
=�[��E����C�O\                                    Bx���L  �          @��?�ff�����\)�)C�H�?�ff�U�L���&��C��
                                    Bx����  �          @��?�����G���z��RC��\?����X���C�
��C���                                    Bx����  
�          @�Q�@
�H��33�~{�d{C��@
�H�R�\�(Q���C���                                    Bx���>  T          @��@녿�����q=qC�3@��J�H�7��ffC�Z�                                    Bx����  �          @�{@	���\(����\�q��C��@	���8���@���p�C�W
                                    Bx��ފ  �          @�  @G��xQ���p��w{C�1�@G��A��A��  C�ٚ                                    Bx���0  "          @��?������z��C��)?��N{�J=q�'�HC�޸                                    Bx����  T          @�?��
�fff��=q=qC�Ǯ?��
�C33�L(��,33C�Y�                                    Bx��
|  T          @��?�=q����=q�wz�C��f?�=q�I���5��p�C�{                                    Bx��"  
�          @�{@�H��33�p���U�\C��q@�H�J�H�����z�C���                                    Bx��'�  "          @�{@�����u��[�C��@��J=q�#33�C�<)                                    Bx��6n  
�          @��R?�Q쿓33���� C���?�Q��N{�?\)���C��{                                    Bx��E  �          @�ff?�(���33���8RC��?�(��P���C33�!��C�q                                    Bx��S�  
�          @�{?�33������\\)C��?�33�N{�Fff�%C��                                    Bx��b`  T          @�p�?�(���  ���(�C���?�(��S33�<(���C���                                    Bx��q  �          @�z�?(��&ff��G�C��H?(��<���`���Ep�C���                                    Bx���  �          @��
?8Q�G����B�C�W
?8Q��B�\�Y���=��C���                                    Bx���R  �          @��
?   �.{��G���C�!H?   �>{�_\)�D�
C��H                                    Bx����  "          @�=q>�G��B�\��
=��C�
>�G��@���Y���@C�,�                                    Bx����  T          @���?녿Q����=C�K�?��B�\�U��<ffC�K�                                    Bx���D  T          @��?
=����p� �
C�4{?
=�.{�aG��Np�C�"�                                    Bx����  
�          @��?+��B�\��{Q�C���?+��?\)�W��>�
C�E                                    Bx��א  �          @��?�����\��ffL�C�5�?���Mp��O\)�3�C�E                                    Bx���6  �          @��>�׿p����L�C�` >���HQ��QG��7z�C�J=                                    Bx����  �          @�=q?fff�z�H���\C�T{?fff�HQ��L(��1z�C��                                    Bx���  "          @��?p�׿L����ff\)C��?p���8���I���7��C���                                    Bx��(  T          @�G�?B�\�Ǯ��p�ǮC���?B�\�a��.{�
=C�\                                    Bx�� �  T          @��?0�׿�=q��p��C�]q?0���p���%��{C�:�                                    Bx��/t  T          @��?@  �����ff�=C��H?@  �s�
�%��Q�C���                                    Bx��>  �          @��?������y���gC��?���z�H�	����p�C���                                    Bx��L�  
�          @�=q?�Q��(Q��>{�2�C�P�?�Q��s�
����z=qC�Y�                                    Bx��[f  
�          @�p�?�\)�7��
=�	�HC��=?�\)�k����H��\)C�y�                                    Bx��j  �          @�33?�  �8Q����
��C���?�  �j�H����љ�C��R                                    Bx��x�  "          @�?�p��$z��Dz��<��C���?�p��s33��G����C��)                                    Bx���X  �          @��?���   �Vff�Zp�C�j=?���]p������\)C�7
                                    Bx����  �          @�?��0  �333�,�\C�~�?��tz�n{�R�HC���                                    Bx����  �          @��?���R�\��\��RC��?���w�    �L��C��{                                    Bx���J  "          @�?�ff�c�
����33C�q�?�ff����>u@HQ�C��                                    Bx����  T          @���@Q��n{��\)��(�C��@Q��x��?J=qA ��C�b�                                    Bx��Ж  T          @�@(��e��B�\�
=C�#�@(��\��?��RA��\C���                                    Bx���<  
Z          @�=q@�R�Z=q�n{�G
=C��)@�R�X��?�  AV�RC��                                    Bx����  
�          @�Q�@=q�Y���J=q�,z�C��f@=q�S�
?�{As
=C��                                    Bx����  
Z          @�Q�@�\�b�\�
=q��Q�C�|)@�\�R�\?�33A��C�o\                                    Bx��.  T          @��@G��l�Ϳ=p��=qC�˅@G��b�\?��A��HC�aH                                    Bx���  �          @���@��q녾�������C��{@��[�?��A�{C�0�                                    Bx��(z  "          @�z�@ ���e�>.{@
�HC��f@ ���=p�@G�A��
C�&f                                    Bx��7   �          @�{@�R�j=q=u?L��C��@�R�E?�(�A��C�j=                                    Bx��E�  
�          @��@0  �W�>u@FffC���@0  �/\)?�(�A֣�C���                                    Bx��Tl  
Z          @�@��u=L��?!G�C��=@��O\)@33A��
C���                                    Bx��c  	�          @�Q�@�\����>W
=@,(�C�w
@�\�S�
@�\A�C��3                                    Bx��q�  �          @�ff@
=�z=q>k�@;�C�+�@
=�Mp�@\)A�{C��                                    Bx���^  
�          @�{@�z=q>�\)@eC�@�L(�@�A��C��H                                    Bx���  �          @�{@��x��?�@�z�C�\@��A�@\)BffC�4{                                    Bx����  �          @�  ?����`  ?�33A��RC��?����\)@AG�B9(�C���                                    Bx���P  �          @��?�p��Tz�?��A���C�˅?�p��ff@:�HB;�C���                                    Bx����  �          @��\?�Q��4z�@�B�C��?�Q쿚�H@Z�HBj�RC�Ff                                    Bx��ɜ  �          @��?���K�������RC�+�?���X��?\)A��C��                                    Bx���B  �          @�
=?�=q�Mp��	�����
C�@ ?�=q�w�����
�HC��H                                    Bx����  �          @�\)?޸R�l�Ϳ����G�C��
?޸R���H>��@��RC��                                    Bx����  �          @���?���p  ��z���Q�C�Ф?����ff>���@���C���                                    Bx��4  �          @�=q?ٙ��p�׿�Q��ɅC�4{?ٙ���
=>��R@vffC�H                                    Bx���  �          @��H?����?\)�E��)�\C���?������Ϳ����Xz�C�XR                                    Bx��!�  �          @��\?˅�1G��R�\�7Q�C���?˅���\��{��  C���                                    Bx��0&  �          @�33?�=q�0  �Tz��933C��3?�=q���\��33���C���                                    Bx��>�  T          @��?�Q��AG��G��'��C���?�Q���ff����W�C��R                                    Bx��Mr  �          @��
?�p��E��G
=�)z�C��q?�p���  ����S\)C���                                    Bx��\  �          @�?���C�
�K��+��C�]q?�����׿�\)�^�RC��                                    Bx��j�  �          @���?�
=�[��(���
��C��?�
=��녾����G�C��f                                    Bx��yd  �          @�(�?�G��U��,���p�C��?�G��������\C�C�                                    Bx���
  �          @��H?�Q��C�
�2�\�\)C�8R?�Q���녿J=q��C���                                    Bx����  �          @��\?�\)�5��E��)(�C��)?�\)���׿�z��k�C�y�                                    Bx���V  �          @��\?�
=�<���Dz��(ffC���?�
=���
����\(�C�                                      Bx����  �          @���?�{�<���A��(p�C�H�?�{���H����Y�C��{                                    Bx��¢  �          @���?���>�R�>�R�$C�o\?�����H��  �L  C��\                                    Bx���H  �          @��@33�<(��3�
�ffC�n@33�~{�^�R�1p�C���                                    Bx����  �          @��\@���7��0  ��HC�R@���xQ�\(��-C��                                    Bx���  �          @�=q@G��5�0���z�C�Q�@G��w
=�aG��2{C�>�                                    Bx���:  
�          @�\)?�p��5�5��=qC�k�?�p��y���s33�Dz�C�s3                                    Bx���  �          @�\)?�p��4z��6ff�\)C��H?�p��x�ÿz�H�I��C�xR                                    Bx���  �          @�=q@��3�
�:=q���C��@��z=q����R�\C�8R                                    Bx��),  �          @��@���1G��9���33C��R@���xQ쿆ff�RffC�!H                                    Bx��7�  �          @�33@�\�/\)�8���C���@�\�u��ff�S�
C�ff                                    Bx��Fx  �          @��H@�� ���A��$�C��{@��mp�������HC�K�                                    Bx��U  �          @��H@!��3�
�&ff�
�C�H@!��o\)�E���C�H                                    Bx��c�  �          @��H@p��0���.{��\C���@p��qG��h���6�RC��
                                    Bx��rj  �          @�G�@&ff�&ff�+���C�~�@&ff�g
=�u�B{C�޸                                    Bx���  �          @���@.�R�%��#�
�	(�C�K�@.�R�a녿\(��/
=C��
                                    Bx����  �          @���@$z��,���#33�	��C���@$z��hQ�J=q�"=qC���                                    Bx���\  �          @��@���9���!��z�C�{@���r�\�+���RC�p�                                    Bx���  �          @�(�@ff�5��3�
�G�C���@ff�xQ�s33�<��C���                                    Bx����  �          @��@��%��AG��$
=C��q@��qG���G��~{C��{                                    Bx���N  �          @�=q@33�p��P  �5��C��f@33�r�\�\����C�:�                                    Bx����  �          @�=q?��H����W
=�=��C��)?��H�qG���z���  C���                                    Bx���  �          @�33@   �   �R�\�7�C�S3@   �u�����z�C���                                    Bx���@  �          @�33?��=q�]p��C  C���?��u��p�����C��3                                    Bx���  �          @�G�?�{���s33�c
=C���?�{�o\)�(����C�                                    Bx���  �          @�=q?���
�H�r�\�`=qC�޸?���s33�����p�C���                                    Bx��"2  �          @�Q�@\)�{�333��C���@\)�c33����k�C��f                                    Bx��0�  �          @���@{�=q�:�H� C���@{�dzῢ�\��=qC�W
                                    Bx��?~  �          @���@���!��9���  C�� @���i����Q��tz�C���                                    Bx��N$  �          @�G�@��{�G
=�-\)C��q@��mp���z����RC�9�                                    Bx��\�  �          @�Q�@�!��Dz��,�C��q@�p  �����G�C���                                    Bx��kp  �          @�  ?�G��\)�S�
�=(�C��?�G��u�������
=C�Z�                                    Bx��z  �          @�  ?�  �{�Tz��>��C���?�  �tz�˅��  C�K�                                    Bx����  �          @��?�\�(��Tz��?(�C��\?�\�s33��{��(�C�xR                                    Bx���b  �          @��R?�ff�
�H�dz��TC��?�ff�l(����H�У�C�g�                                    Bx���  �          @��R?�Q����c33�S�C���?�Q��fff�   ��G�C���                                    Bx����  �          @��?�z����`���SG�C�Y�?�z��dz���H��\)C�y�                                    Bx���T  �          @�p�?�G��G��aG��RC�y�?�G��a녿��R�֏\C�7
                                    Bx����  �          @��?�p���=q�[��O(�C���?�p��Tz�� ����(�C�aH                                    Bx���  �          @�(�@녿�Q��_\)�Sp�C�
@��O\)�	����z�C��                                    Bx���F  �          @�(�?�녿��e��[z�C�:�?���P���\)��\)C�f                                    Bx����  �          @���?�  ��ff�g��\��C��?�  �X��������
C��f                                    Bx���  �          @�z�?��H��\�hQ��_z�C���?��H�W���R��{C�k�                                    Bx��8  �          @��?�33���
�[��SQ�C�b�?�33�Q��33��C�                                    Bx��)�  �          @��?��
�У��^�R�\�RC�� ?��
�J�H�
�H��Q�C���                                    Bx��8�  �          @�
=?˅��Q��`���b=qC���?˅�O\)�
�H���\C��                                    Bx��G*  �          @��R@녿��
�]p��\�C���@��6ff����C���                                    Bx��U�  �          @�\)@�
���
�X���U
=C��q@�
�&ff����
ffC�˅                                    Bx��dv  �          @�
=@��u�S�
�NC�C�@��   ����
C�q                                    Bx��s  �          @��R@  �u�Z�H�Y�RC�|)@  �#33�{��RC���                                    Bx����  �          @��R@p��xQ��\(��[��C��@p��%���R�\)C�Q�                                    Bx���h  �          @��?�
=�J=q�e�oz�C��?�
=�\)�,���"p�C��                                     Bx���  �          @�@
=q�aG��^{�`�RC��@
=q� ���#�
���C�T{                                    Bx����  �          @�  @z�s33�[��W�C���@z��#33��R���C�'�                                    Bx���Z  �          @�p�@z�+��X���Y�HC���@z����%��HC��H                                    Bx���   �          @�{@�
�0���e��j�HC��=@�
����0  �#�C�^�                                    Bx��٦  �          @�G�@������aG��\�
C��@���
�8Q��(��C��
                                    Bx���L  �          @�Q�@�H�\)�`  �]
=C�g�@�H����=p��/�C�                                      Bx����  �          @�@33>B�\�]p��a
=@�G�@33�����E�?�RC��R                                    Bx���  �          @�  @�\?����B�\�F�\Aݮ@�\���H�N�R�Xz�C���                                    Bx��>  T          @�  ?�z�?���S�
�b(�A癚?�z�=p��X���kQ�C�`                                     Bx��"�  �          @��?��H?!G��n{��A��H?��H�����b�\�s�\C��                                    Bx��1�  �          @\)?�\>aG��`���|Q�@�?�\�Ǯ�I���T��C�AH                                    Bx��@0  �          @xQ�?�?xQ��Q��g�A�=q?��L���Tz��l��C�#�                                    Bx��N�  �          @tz�?��?Tz��N{�e�RA�(�?�׿c�
�L���c�
C�Q�                                    Bx��]|  �          @g
=?�p�>���Mp��qA  ?�p������:=q�\ffC���                                    Bx��l"  �          @Q�?�R�aG��<(�C��R?�R��R�
=�)\)C��{                                    Bx��z�  �          @�?Tz�\)� ���C�H�?Tz῏\)���H�\C�B�                                    Bx���n  �          @��?���>#�
� ���tp�A?��Ϳ\(������R=qC���                                    Bx���  �          @+�?��\���Q�aHC�p�?��\���Ϳ�=q�/��C�@                                     Bx����  �          @1G�?�
=�c�
���dp�C�u�?�
=��z����	�RC��
                                    Bx���`  �          @4z�?�p��Y������M�C�?�p�����
=��(�C��\                                    Bx���  �          @)��?��H�}p���TG�C�c�?��H��33�������C�Q�                                    Bx��Ҭ  �          @ff?n{�O\)���e=qC�j=?n{��zῡG���\C���                                    Bx���R  �          @N�R��\)�8�þ�{����Ct����\)�*�H?��A�=qCr�3                                    Bx����  �          @>{���\�&ff����:�\Cs�R���\�#�
?E�AqCs�\                                    Bx����  �          @9���n{�#�
�s33��z�Cz\�n{�,��>�G�A  C{                                      Bx��D  �          @Y���E��>�R��z����C���E��J�H>��A{C�4{                                    Bx���  �          @q녿����X�ÿ��H��z�C|�������b�\?�RA�C}@                                     Bx��*�  �          @s�
�fff�S33��z���
=C~��fff�l(�>B�\@5�C�"�                                    Bx��96  �          @tz�0���L(����R����C��0���p�׾#�
�z�C��{                                    Bx��G�  �          @p�׿��2�\���"ffC��f���hQ�=p��6�HC��q                                    Bx��V�  �          @k�=L���$z��%�4��C��3=L���`�׿z�H�z{C�k�                                    Bx��e(  �          @l��?
=q���R�A��_�C���?
=q�N�R��z���G�C��=                                    Bx��s�  �          @o\)?�  ��(��@���VC�q�?�  �L�Ϳ�33�ѮC��\                                    Bx���t  �          @n{?�{��ff�C�
�]p�C���?�{�Dz���
���C��                                    Bx���  �          @o\)?�  ��p��J�H�j
=C�{?�  �6ff�����C��)                                    Bx����  �          @p  ?�G��ٙ��Fff�^�
C�P�?�G��@  ��\)����C�g�                                    Bx���f  �          @w�?���=q�N�R�bz�C��
?��=p���\���\C�Ф                                    Bx���  �          @w�?�(������N{�a33C���?�(��<(���\��z�C�AH                                    Bx��˲  �          @w
=?�����p��G��X�RC��q?����A녿����C��H                                    Bx���X  �          @��\?�(���\�L���O��C��?�(��U��ff���C��                                     Bx����  �          @�ff?�G���
=�g��o��C�|)?�G��N�R��
=C��=                                    Bx����  �          @��R?��
��33�n�R�z�C��3?��
�P���p���RC��)                                    Bx��J  �          @�  ?+����h���n(�C��?+��c33�(���p�C�P�                                    Bx���  �          @���?=p�����\(��`  C�W
?=p��fff��
=��(�C��
                                    Bx��#�  �          @��H?B�\�G��^{�h��C�Y�?B�\�\���33��
=C�9�                                    Bx��2<  �          @���?�p��G��E�H\)C�@ ?�p��^�R�˅����C��)                                    Bx��@�  �          @��\?�  �����P  �S��C���?�  �Q녿�33��C�G�                                    Bx��O�  �          @��?�{��z��K��H=qC�\?�{�N{��{��=qC��)                                    Bx��^.  �          @��?�������e��g�RC�o\?���:�H��R�p�C���                                    Bx��l�  �          @��R@z��ff�*�H�!��C��=@z��Vff��Q����HC���                                    Bx��{z  �          @��
@�H�S�
�Q���33C��@�H�{�����L��C��3                                    Bx���   �          @��@��L���{����C�B�@��xQ������
=C���                                    Bx����  �          @���@{�R�\��{���C�w
@{�q녽#�
�\)C��
                                    Bx���l  �          @��H@   �X�ÿ�����C�<)@   �vff<��
>��C��                                     Bx���  �          @�  @���Vff������C�Ф@���s33=L��?+�C�(�                                    Bx��ĸ  �          @�{@\)�Vff��{��ffC��=@\)�u����
��  C�.                                    Bx���^  �          @�p�@(��\(���
=���RC�4{@(��u�>B�\@(�C�޸                                    Bx���  �          @�p�@�\�]p��\����C���@�\�qG�>���@�G�C���                                    Bx���  �          @�(�@ff�Vff������HC���@ff�k�>��@X��C�G�                                    Bx���P  �          @��H@
=�P  ��33��(�C���@
=�h��=�?\C�z�                                    Bx� �  �          @�33@�H�Mp���z���C��H@�H�g�=��
?��
C��                                    Bx� �  �          @��@#33�P�׿�  ��
=C�H@#33�e�>�  @N{C���                                    Bx� +B  �          @�
=@���Z=q������
C���@���q�>B�\@p�C�'�                                    Bx� 9�  �          @�
=@=q�Z=q�˅����C��R@=q�p��>k�@?\)C�Q�                                    Bx� H�  �          @��
@�Vff��ff��Q�C�� @�k�>k�@Dz�C�=q                                    Bx� W4  �          @�ff@���Q녿�G��_\)C��@���W�?.{A�C��\                                    Bx� e�  �          @�p�@��U����
�e�C�7
@��Z�H?.{Ap�C��)                                    Bx� t�  T          @�
=@��S33�}p��Z{C�%@��XQ�?333A(�C��
                                    Bx� �&  �          @��H@(��aG��.{���C�]q@(��Z�H?���Ae�C��                                     Bx� ��  �          @}p�?�  �j�H�\)� ��C�&f?�  �R�\?�\)A���C�E                                    Bx� �r  �          @���@1��P  ��
=��{C�C�@1��b�\>�\)@dz�C��                                    Bx� �  �          @���@>�R�@�׿������C�Z�@>�R�[����
���C�~�                                    Bx� ��  �          @�33@<(��L(������ffC�Z�@<(��a�>��?��C��H                                    Bx� �d  �          @���@?\)�Dzῷ
=���
C�"�@?\)�XQ�>B�\@��C��                                     Bx� �
  �          @�33@<���B�\�h���B{C�
=@<���G
=?!G�A�RC��R                                    Bx� �  �          @��@8Q��QG�������C���@8Q��J=q?�G�AU��C�#�                                    Bx� �V  �          @�@A��J=q�&ff�33C��f@A��E?h��A>�HC�4{                                    Bx��  �          @��@G
=�J=q�333�(�C�>�@G
=�G�?^�RA2ffC�o\                                    Bx��  �          @�\)@AG��P  ��33���C�q�@AG��A�?�(�A}��C�l�                                    Bx�$H  �          @���@7��]p�>�@��C���@7��5@�A��C��                                    Bx�2�  �          @���@.{�P  ?�z�Aup�C���@.{��@#33B��C��)                                    Bx�A�  �          @�z�@7
=�R�\?�@�{C�|)@7
=�*=q@ ��A�  C���                                    Bx�P:  �          @��@8Q��QG�?���A\��C��=@8Q����@�RB  C�#�                                    Bx�^�  �          @�  @=p��S33?(��A�HC��3@=p��'
=@Q�A�=qC�N                                    Bx�m�  �          @�Q�@Fff�O\)>�@���C��@Fff�(��?�Q�A�C��\                                    Bx�|,  �          @�Q�@K��L��>\)?�=qC�l�@K��0��?�\)A�(�C��                                    Bx���  �          @�@O\)�Vff�\���RC�  @O\)�I��?�(�ArffC���                                    Bx��x  �          @��@S�
�P  ��������C���@S�
�C�
?�z�Af�RC���                                    Bx��  �          @�p�@XQ��L�;�z��e�C�G�@XQ��>{?�p�At��C�Y�                                    Bx���  �          @���@W��K���p����\C�S3@W��?\)?�33Ad��C�5�                                    Bx��j  T          @��R@\(��L(���G����\C���@\(��A�?��AV{C�U�                                    Bx��  �          @��R@c33�E����
�z=qC��f@c33�8Q�?��A`z�C�z�                                    Bx��  �          @�  @j�H�@�׾L���{C�W
@j�H�0��?��HAl  C���                                    Bx��\  �          @�  @l(��>�R��\)�UC���@l(��1G�?���A\��C��                                    Bx�   �          @�  @n{�;���p�����C��f@n{�1G�?��\AF=qC��=                                    Bx��  �          @���@mp��@  �\)��C���@mp��:�H?c�
A)�C��                                    Bx�N  �          @�G�@k��>�R�(����z�C���@k��<��?J=qA  C���                                    Bx�+�  �          @��\@l(��@�׿L����
C�j=@l(��B�\?.{A ��C�G�                                    Bx�:�  �          @��\@r�\�:�H�(���\)C�:�@r�\�7�?L��A��C�t{                                    Bx�I@  �          @��
@s�
�;��8Q����C�/\@s�
�<(�?333Ap�C�*=                                    Bx�W�  �          @�G�@P  �J=q��Q���G�C��@P  �l(���\)�N{C��\                                    Bx�f�  �          @�(�@.�R�S33�3�
�  C���@.�R��
=�}p��1C�w
                                    Bx�u2  �          @�z�@*=q�U��7
=�p�C�J=@*=q���׿��\�7\)C��3                                    Bx���  �          @�(�@4z��Z=q�%��\C���@4z����R�@  �C��                                    Bx��~  �          @�@<(��W��&ff���
C���@<(����G��33C���                                    Bx��$  �          @�\)@C33�_\)�����C��{@C33���R������C��{                                    Bx���  �          @�
=@E��^{�Q���p�C�@E����
=q���RC�.                                    Bx��p  �          @�@C�
�Z�H�����ffC��@C�
��z�z���{C�E                                    Bx��  �          @��
@E��W
=������C�E@E������\)�ǮC��                                     Bx�ۼ  �          @�(�@E�\���������C��@E��=q�Ǯ����C���                                    Bx��b  �          @�p�@C�
�[��
=��C��@C�
��(��\)��ffC�L�                                    Bx��  �          @��@AG��W�����ԸRC��=@AG����ÿ   ��(�C�h�                                    Bx��  �          @�33@<���a����ʣ�C��@<����zᾳ33�{�C��H                                    Bx�T  �          @��@AG��c�
����=qC�&f@AG����
��  �5�C�!H                                    Bx�$�  �          @���@C�
�]p����H���C���@C�
�~{�B�\�	��C�˅                                    Bx�3�  �          @��@_\)�'���R��z�C���@_\)�U�O\)��C�!H                                    Bx�BF  T          @�(�@mp��p���p���=qC�5�@mp��E�+����
C�
                                    Bx�P�  �          @��
@c�
�6ff��(���\)C���@c�
�Tzᾅ��C33C��                                     Bx�_�  �          @�z�@fff�0  ����{C�O\@fff�QG��\����C��H                                    Bx�n8  �          @��\@^{�333������z�C���@^{�U��������C��                                    Bx�|�  �          @��@W��HQ�ٙ���z�C��@W��c�
������C���                                    Bx���  �          @�{@W��QG���G���z�C��=@W��fff=�G�?��
C���                                    Bx��*  �          @�\)@Z=q�[���33�UG�C�p�@Z=q�e�?   @�G�C���                                    Bx���  �          @�Q�@^�R�Z�H����R�RC��q@^�R�dz�?   @�
=C�                                      Bx��v  �          @��@b�\�Z�H��z��T��C��@b�\�e�>�@���C�Y�                                    Bx��  �          @�G�@fff�Q녿��\�h��C��@fff�`��>��
@l(�C��                                     Bx���  �          @���@z=q������֣�C��@z=q�,�Ϳ�\)�T��C��\                                    Bx��h  �          @�ff@xQ�c�
�8����HC��f@xQ����(���{C���                                    Bx��  �          @��
@j�H�O\)�C�
��\C��q@j�H���Q���C�5�                                    Bx� �  �          @�z�@p  ��Q��Fff���C�G�@p  ��Q��'���C��H                                    Bx�Z  �          @�(�@qG��\�C33�G�C�#�@qG���
=�$z���G�C���                                    Bx�   �          @�33@n{�(��C33�
=C�U�@n{����{��33C���                                    Bx�,�  �          @�=q@i�������G��!ffC��R@i����p��'��  C�K�                                    Bx�;L  �          @�(�@e��8Q��S�
�*�C��=@e���\)�9����\C��R                                    Bx�I�  �          @�z�@`  ���
�Y���0(�C�e@`  ���
�:�H�ffC��=                                    Bx�X�  �          @�@`�׾��R�\(��133C�y�@`�׿��
�<����\C��f                                    Bx�g>  �          @�z�@l(������<(���C���@l(����
=q��{C�&f                                    Bx�u�  �          @��R@e����-p��G�C���@e��ÿ�{��Q�C�&f                                    Bx���  �          @�{@g���(��-p��z�C��@g��녿�z���Q�C���                                    Bx��0  �          @�
=@z�H���\�1G����C��)@z�H�ff��Q���{C�}q                                    Bx���  �          @�@}p���z��%���ffC�8R@}p���ÿ��H���\C�s3                                    Bx��|  �          @�z�@|�Ϳ�ff����\C�L�@|����Ϳ\���RC�R                                    Bx��"  �          @���@{��\�\)��{C�g�@{���Ϳ�=q���\C��                                    Bx���  �          @�(�@w��У���R��=qC��3@w��#33���
��  C�Y�                                    Bx��n  �          @�z�@z�H�\��R��(�C�c�@z�H�(���=q���HC��                                    Bx��  �          @��
@}p���z��p����C�,�@}p����{��=qC���                                    Bx���  �          @��
@~�R���R� ������C�U�@~�R��Ϳ�p�����C���                                    Bx�`  �          @��@y�������(�����C�xR@y���{��\)����C�1�                                    Bx�  �          @��@�  ��G�����ޣ�C���@�  ���������C��                                     Bx�%�  �          @��
@~�R�У���\��z�C�� @~�R��Ϳ�\)����C�+�                                    Bx�4R  �          @���@\)���H�G����HC�` @\)�!G������y��C�ٚ                                    Bx�B�  �          @���@��׿�  ����G�C�4{@����!G���(��fffC��
                                    Bx�Q�  �          @��\@q녿�p��.{�	�RC�H@q��G���Q���C�|)                                    Bx�`D  
�          @���@g��h���>�R�Q�C���@g���33����C���                                    Bx�n�  �          @��
@b�\�G��7��p�C��f@b�\��
=�����z�C��{                                    Bx�}�  �          @�33@^{�z��=p�� Q�C�@ @^{���
�����C�c�                                    Bx��6  �          @��\@XQ�p���<��� (�C�:�@XQ��ff�G���C�\                                    Bx���  �          @��\@Y���^�R�<(��C���@Y����\��\��(�C��f                                    Bx���  �          @��H@\�Ϳc�
�8���\)C��@\�����  ��\C�Ǯ                                    Bx��(  �          @��H@^{�G��9����HC���@^{��Q��33��ffC�g�                                    Bx���  �          @��@Y���k��C33�#z�C�g�@Y��������
=C�H                                    Bx��t  �          @��@Y���h���C�
�$
=C���@Y���
=�������C�{                                    Bx��  "          @��@XQ�c�
�?\)�"G�C���@XQ���
����RC�L�                                    Bx���  �          @��H@_\)�W
=�5��C�8R@_\)���H��R����C�W
                                    Bx�f  �          @��
@l(��(���,����C��@l(���p�����RC�l�                                    Bx�  �          @��H@l(��(��)���{C�S3@l(���z��	���ߙ�C���                                    Bx��  �          @���@hQ��\�,(��{C��@hQ�˅��R����C�0�                                    Bx�-X  �          @�G�@i���z��)����C�}q@i���У��
=q��p�C���                                    Bx�;�  �          @���@l�Ϳ\)�%�
��C��3@l�Ϳ˅�����C�c�                                    Bx�J�  �          @��H@j=q�8Q��*=q�C�xR@j=q��G�����=qC�0�                                    Bx�YJ  �          @��@n{�&ff�(Q��33C�
=@n{��Q�����33C��3                                    Bx�g�  �          @���@i���!G��'
=���C��@i����z�����z�C��                                    Bx�v�  �          @���@j�H�\)�(Q��ffC���@j�H�����
=q����C�1�                                    Bx��<  �          @��@hQ����%��RC��{@hQ�����Q���ffC�N                                    Bx���  �          @��\@dz�+��1����C���@dz��G������{C��H                                    Bx���  �          @���@p  ���
�   ��C��@p  �����	������C�E                                    Bx��.  �          @��@xQ�=��������\)?�p�@xQ�c�
�{��\)C��H                                    Bx���  �          @�{@q�>Ǯ��\����@�ff@q녿z��  ��C���                                    Bx��z  �          @�p�@k�>W
=�=q�@O\)@k��L����\���
C�޸                                    Bx��   �          @�p�@l��=L���Q����?L��@l�Ϳn{�(�����C��{                                    Bx���  T          @��H@u�?k���\��33AX  @u�=����R��\)?޸R                                    Bx��l  �          @�
=@}p�?333��
=��33A (�@}p�����33�ڣ�C���                                    Bx�	  �          @�p�@p  ?z������ffAG�@p  �Ǯ�33��\)C�f                                    Bx��  �          @��@p��?8Q�����z�A,z�@p�׾�����H��C�                                    Bx�&^  �          @���@p��?h������Q�AYG�@p�׽��   �G�C�)                                    Bx�5  �          @���@u�?�G��p�����Al  @u�<����ff>Ǯ                                    Bx�C�  �          @�G�@p��?�  �����
Ao\)@p�׼��"�\�Q�C�Ǯ                                    Bx�RP  �          @��@h��?��������HA
=@h��=#�
�\)�	p�?\)                                    Bx�`�  �          @��
@e�?�=q�����\)A�Q�@e�=�\)�   ��?�Q�                                    Bx�o�  �          @���@_\)?�ff��� 33A��@_\)>�  �*=q��H@��H                                    Bx�~B  T          @�p�@e�>�{�
=�=q@��R@e��#�
�33�G�C���                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���  �          @���@7�?����\���B{@7�?fff�'��$ffA�                                     Bx��4  �          @��R@E���  �4z��)(�C��\@E���{��R�ffC�!H                                   Bx���  �          @��@>{�Tz��9���-{C�0�@>{���z��ffC���                                    Bx�ǀ  �          @�\)@K��aG��(����C�B�@K���{�z���=qC���                                    Bx��&  �          @��@B�\�u�'
=�(�C�=q@B�\��� ���陚C��)                                    Bx���  �          @~{@<�Ϳz�H�=q�=qC��f@<�Ϳ�{�����ܣ�C��)                                    Bx��r  �          @���@C�
��{�p��z�C��@C�
�{��(����C���                                    Bx�  �          @�=q@A녿�  �!G���HC���@A녿���z���Q�C��R                                    Bx��  �          @��@E���\)�'�� �HC�aH@E���ff��\�	�C��R                                    Bx�d  �          @q�@8�ÿ(�����C�H@8�ÿ��R��z���\)C�]q                                    Bx�.
  �          @\��@\)��
=�z��\)C�Q�@\)��33��
=��z�C�T{                                    Bx�<�  T          @r�\@*�H�W
=��H�#{C�N@*�H��(�������
C��)                                    Bx�KV  �          @��
@<(��\)�5��-�HC���@<(��У������C�|)                                    Bx�Y�  �          @�p�@@  �333�3�
�)��C�p�@@  �޸R��
���C���                                    Bx�h�  �          @��@J�H�\)�<(��)��C��
@J�H����R�
�C��                                    Bx�wH  �          @��R@C�
�fff�0  �#33C��{@C�
��33�����C��                                    Bx���  �          @�z�@<�Ϳ����*�H�!�C��H@<����
�����C��                                    Bx���  �          @�ff@<�Ϳs33�4z��(��C��@<�Ϳ�p��{���C�)                                    Bx��:  �          @�(�@;��}p��.�R�%�C���@;���p��Q����HC���                                    Bx���  �          @�
=@DzῡG��&ff�G�C�Ф@Dz��
�H��33����C�XR                                    Bx���  �          @�=q@9�����p��33C��
@9���G���(���{C�                                    Bx��,  �          @�G�@7���z���
�G�C���@7�����p���Q�C���                                    Bx���  �          @���@.{��33�\)�{C�j=@.{��R��z���C��{                                    Bx��x  �          @��@.{��\�=q�p�C�q�@.{�$z������C�S3                                    Bx��  T          @�z�@.�R���R�����C��
@.�R�0�׿�Q�����C�]q                                    Bx�		�  �          @�  @ ���!��33�G�C�XR@ ���N{��33�{
=C��\                                    Bx�	j  �          @�?�p��H��������C��?�p��tz῁G��S�C��=                                    Bx�	'  �          @�@33�A���R�G�C�
=@33�p  �����k�
C�W
                                    Bx�	5�  
�          @�=q@(��<���\)���C�L�@(��e��s33�K33C��
                                    Bx�	D\  T          @��?�Q��S�
�
�H���
C�&f?�Q��x�ÿ=p��ffC�9�                                    Bx�	S  �          @��
@���H���
=��ffC�(�@���mp��@  ��
C��                                    Bx�	a�  �          @�33@ ���L(��
�H��p�C�
@ ���q녿J=q�'�
C���                                    Bx�	pN  �          @��@   �"�\���R��C�@ @   �G
=�aG��F�\C�c�                                    Bx�	~�  �          @�33?��R�c33��\�ߙ�C�Y�?��R��녿
=q����C��                                    Bx�	��  �          @�(�?�  �Z�H������C���?�  �{��z����C��)                                    Bx�	�@  �          @�G�?����e��=q��Q�C���?������׾�{��  C���                                    Bx�	��  �          @���?�G��mp�����
=C��?�G���(����R����C��                                    Bx�	��  �          @���?����mp��������C�w
?�����(����
���RC���                                    Bx�	�2  �          @�Q�?u�u������
=C��?u��z�#�
���C��=                                    Bx�	��  �          @��
?z�H�_\)�����(�C��)?z�H�|�;�G���{C���                                    Bx�	�~  �          @�ff>��}p����H����C�H�>����
>�p�@�G�C�(�                                    Bx�	�$  T          @�(�?��P  ����Q�C���?��l(����θRC�/\                                    Bx�
�  T          @�z�?���hQ쿹����
=C��=?���z=q<#�
=���C�AH                                    Bx�
p  �          @��@ff�U���\�h  C�
@ff�^�R>���@�C���                                    Bx�
   �          @���@\)�*�H��
=��  C���@\)�L�ͿL���9G�C�~�                                    Bx�
.�  �          @�G�@(��#33��R��C�Ff@(��L�Ϳ�\)��
=C�+�                                    Bx�
=b  �          @�=q@!��7
=��ff����C�@!��H�þ���	��C�q�                                    Bx�
L  T          @��@33�*�H�(���
=C�^�@33�R�\��ff�j=qC�y�                                    Bx�
Z�  �          @�  @\)�4z����� �C�AH@\)�\�Ϳ���g33C�~�                                    Bx�
iT  �          @���@p��G
=��\)��Q�C���@p��e��
=��{C��)                                    Bx�
w�  �          @��R@1G��
=���C�\)@1G��4zῳ33���\C�9�                                    Bx�
��  �          @�
=@:=q��33�!����C�33@:=q�{��  ��33C��R                                    Bx�
�F  �          @�
=@<(������"�\�C��{@<(������
��  C�B�                                    Bx�
��  �          @�33@?\)����$z��(�C�<)@?\)��
=�   ��\)C��
                                    Bx�
��  �          @y��@>{?^�R���A�Q�@>{�L���   � ffC��f                                    Bx�
�8  �          @��\@.{�8Q��;��8z�C���@.{��p��p��z�C��H                                    Bx�
��  �          @��\@%����
�7
=�3(�C���@%���R����{C���                                    Bx�
ބ  �          @�(�@ff���6ff�.��C�Y�@ff�@�׿����33C�p�                                    Bx�
�*  �          @�z�@������@  �:33C�33@��#�
��R�\)C�j=                                    Bx�
��  T          @y��@(�ÿ#�
�2�\�7(�C�/\@(�ÿ�{���  C�U�                                    Bx�
v  �          @u�@5�J=q�(����C�>�@5��\)�   ��\)C�(�                                    Bx�  �          @�  @;�=����,���*p�?�p�@;��aG��#33�\)C���                                    Bx�'�  �          @xQ�@*�H?=p��'��-�
Avff@*�H����-p��5G�C�<)                                    Bx�6h  �          @y��@!�?p���3�
�8�RA�Q�@!녾\)�=p��E�RC�t{                                    Bx�E  �          @u�@!�?���*=q�0{A�G�@!�=�\)�7��B?\                                    Bx�S�  �          @w
=@��?��R�.�R�5(�B��@��>���E��UffA&ff                                    Bx�bZ  �          @z�H@-p�?���%�%p�A��R@-p�>\)�4z��8��@>�R                                    Bx�q   �          @z�H@p�?����,���-p�A�{@p�>Ǯ�B�\�J�AG�                                    Bx��  �          @n{?��R?��H�(Q��4�B"�H?��R?(���Dz��^��A�33                                    Bx��L  �          @n�R@p�?��
�=q�!ffB��@p�?L���8Q��K�A���                                    Bx���  �          @r�\@+�?(���!G��)��A\z�@+���\)�%�/p�C�f                                    Bx���  �          @vff@8��=�\)�"�\�%=q?���@8�ÿW
=����Q�C���                                    Bx��>  �          @s�
@0�׽����#�
�+(�C���@0�׿}p��
=�(�C��                                    Bx���  �          @p  @%�>u�,���8�R@�p�@%��8Q��'
=�1�C�+�                                    Bx�׊  �          @w
=@1�?
=�'
=�*�A?�
@1녾�Q��)���-p�C�C�                                    Bx��0  �          @w
=@@��    ��ff<��
@@�׿Q��(��G�C�^�                                    Bx���  �          @u�@8Q�?
=q�����A)@8Q쾨���=q�  C��                                    Bx�|  �          @l��@�@�\�Q��{B,�H@�?��-p��>�A�{                                    Bx�"  �          @h��?�p�@����z��Q�BX�R?�p�?�=q�'��A  B)��                                    Bx� �  �          @e?�Q�@G��"�\�=G�Bn  ?�Q�?�G��E��|�
B ��                                    Bx�/n  �          @u@33?u�-p��=��A���@33�L���8Q��MG�C�o\                                    Bx�>  �          @��R@�@�
�#33�G�B.(�@�?��
�L(��Ep�A��                                    Bx�L�  �          @�  @�@���'
=�B7�\@�?����Q��J�A�R                                    Bx�[`  �          @�
=@{@��"�\�{B*��@{?��\�J�H�C(�A���                                    Bx�j  �          @�ff@$z�@
=q�p����B Q�@$z�?�
=�C�
�<A�p�                                    Bx�x�  �          @�33@(�@{�=q��B){@(�?�  �A��?Q�A�33                                    Bx��R  �          @���@�H?���#�
���B@�H?c�
�C33�Gz�A�G�                                    Bx���  �          @w�@��@G��{�Q�B*
=@��?�ff�@���L  A˙�                                    Bx���  �          @tz�@%�?����R�!�A�(�@%�>���2�\�;p�A                                    Bx��D  �          @u�@p�?�(��%��,A�33@p�>�=q�5�C@��                                    Bx���  �          @l(�@G���\�
�H�+�C���@G���(���{�33C��q                                    Bx�А  �          @l(�@�R��ÿУ���=qC���@�R�333�8Q��5C�B�                                    Bx��6  �          @k�@
=��G��&ff�4�C�C�@
=�33��
=��C�K�                                    Bx���  �          @dz�@
=�Tz��ff�,C�\)@
=���ÿ�
=�C�(�                                    Bx���  �          @��\?n{�{�?��A��RC��?n{�K�@1�B   C�0�                                    Bx�(  �          @�?=p��g
=?�z�A�p�C��\?=p��0  @AG�B:�RC��                                    Bx��  �          @�  ?��r�\?�Aʣ�C�  ?��=p�@?\)B3G�C�)                                    Bx�(t  �          @�  ?Y���i��?�(�Aݙ�C��3?Y���1G�@E�B:�C���                                    Bx�7  �          @�
=?aG��mp�?��A���C��{?aG��8��@;�B0\)C��H                                    Bx�E�  T          @���?aG��fff?�=qA�ffC��)?aG��7�@,��B'�C���                                    Bx�Tf  �          @�(�?Y���U�@�BG�C�!H?Y���Q�@QG�BQQ�C�Ф                                    Bx�c  �          @Y��?(��,(�?�Q�B�\C�XR?(����@.�RBX{C��f                                    Bx�q�  �          @I�����
�@��?J=qAk�C�Ǯ���
�$z�?�  B�HC���                                    Bx��X  �          @Fff<#�
�333?Q�A��HC�{<#�
��?��HBG�C�R                                    Bx���  �          @(��>����?��HB=qC�E>�녿���@ffBQ(�C�<)                                    Bx���  �          @.�R>.{��?�z�B2�HC���>.{��
=@�B�HC�:�                                    Bx��J  �          ?�\)�aG��fff?�\)Bp�
C�AH�aG����R?�=qB�k�Ckz�                                    Bx���  �          ?�\)���þ\)?xQ�B��qCK�����>B�\?uB�\C�                                    Bx�ɖ  �          @@�׿(�@(Q�?�\)A�B�  �(�@;�>�(�AffB�\)                                    Bx��<  T          @K��&ff@:=q?��A�ffB�LͿ&ff@Fff=#�
?B�\B���                                    Bx���  �          @H�ÿ��@B�\>�G�A ��B�(����@@�׿
=�,��B�W
                                    Bx���  �          @N{�G�@Fff>�33@���B�k��G�@B�\�.{�Dz�B��                                    Bx�.  �          @N�R��\)@@��>B�\@W�B�녿�\)@:=q�O\)�h(�B�B�                                    Bx��  �          @H�ÿ�\)@7
=>�
=@�\)B޳3��\)@5�
=q� ��B���                                    Bx�!z  �          @1G�����@
=>��A
=B�{����@
=�\�Q�B���                                    Bx�0   �          @-p����H@녾��H�9��B�R���H?�G���\)��
=B��                                    Bx�>�  �          @)����G�@�
������B����G�?��Ϳ�
=�Q�C Y�                                    Bx�Ml  �          @4zΐ33@Q쿹������B��)��33?����
�;�HB��                                    Bx�\  �          @1녿@  @G���  �{B܏\�@  ?�����
�a
=B�                                      Bx�j�  �          @*�H�+�?��Ϳ���)33B��ÿ+�?���\�m�B�3                                    Bx�y^  �          @5��Y��?���(��L��B��f�Y��?@  �$z�B�C
�\                                   Bx��  �          @=u?@  ��p�{B�=q=u>�����ª�B���                                   Bx���  �          @8�ÿW
=@�R?!G�AX(�B٣׿W
=@#33�L����Q�Bخ                                    Bx��P  �          @:=q����@��?�ffA�33B噚����@*=q>L��@�G�B�.                                    Bx���  �          @>�R����@!�?z�HA���B�  ����@-p�=�@Q�B���                                    Bx�  �          @G�����@3�
>��
@�Q�B�녿���@0�׿
=�4(�B��{                                    Bx��B  �          @L�Ϳ��@8�þL���b�\B�3���@+���{��z�B�#�                                    Bx���  �          @*=q��G�@p����H  B�aH��G�?����R���HB��                                    Bx��  �          @J�H���R@0�׾k����B�q���R@#33������33B�                                    Bx��4  �          @4zῬ��?�z�:�H���HB�� ����?�=q��ff�   C��                                    Bx��  �          @#�
����?��H��(��8p�C�����?���G��f(�C�3                                    Bx��  �          @6ff��Q�?��H�(��F�\C����Q�?���Q���=qCB�                                    Bx�)&  �          @(�ÿ��
@�\>�{@���B�uÿ��
@녾����
ffB��                                    Bx�7�  �          @,(���(�@p��u��{B�8R��(�@G��k���=qB���                                    Bx�Fr  �          @����H?������|(�B�33���H?�33�333��p�B��                                     Bx�U  �          @3�
��ff@
=>u@�(�Cn��ff@���G��Q�C޸                                    Bx�c�  �          @;���\?ٙ�?���BQ�C(���\@ff?uA�C\                                    Bx�rd  T          @-p����@\)?n{A��B�����@�H>B�\@��\B�aH                                    Bx��
  �          @8�ÿ��
@=q?uA���B��Ϳ��
@%>#�
@H��B�                                     Bx���  �          @<(����@�
?�33A�p�B�����@#33>\@�{B�(�                                    Bx��V  �          @.�R�c�
@��>�Q�A ��Bܮ�c�
@�þ����ffB�Ǯ                                    Bx���  �          @p��z�?�����S
=BՅ�z�?�
=�������B���                                    Bx���  �          @33���R@�Ϳ   �K\)B�#׾��R?�Q쿔z���{B�G�                                    Bx��H  �          @5���R@,(��
=q�2{B����R@�ÿ������B�=q                                    Bx���  �          @3�
�W
=@)��<��
>�p�B�8R�W
=@!녿L�����B���                                    Bx��  �          @L�Ϳ�{@:�H>��@�Bݔ{��{@7
=�&ff�@��B�k�                                    Bx��:  �          @dz��33@0��?��
A�(�B�Ǯ��33@Fff?(�A
=B��                                    Bx��  �          @c33���\@I��?+�A333B��
���\@Mp����
���
B�(�                                    Bx��  �          @n{?�{?333�H���f�A��?�{��z��L���n=qC��=                                    Bx�",  �          @j�H?�\?
=�G��l(�A��\?�\�Ǯ�I���p  C�˅                                    Bx�0�  �          @y��?�\)?�
=�P  �ap�BKG�?�\)?(��g
=�A��
                                    Bx�?x  �          @w
=?�
=?�
=�B�\�O��B3�H?�
=?.{�Y���w�
A��                                    Bx�N  �          @vff?�  @�
�7
=�A33BW\)?�  ?���Vff�s�HB{                                    Bx�\�  �          @y��?���@���-p��0��Bv�?���?\�Tz��kG�BD
=                                    Bx�kj  �          @��?h��@:=q�0���(B�(�?h��?�Q��`���jz�B��{                                    Bx�z  �          @��H?=p�@<���/\)�'��B���?=p�?��R�`  �k  B�\)                                    Bx���  �          @s33?!G�@n{�\���RB�?!G�@\(���p����B�33                                    Bx��\  �          @e�>��H@^�R�#�
�%�B���>��H@HQ��z���33B�                                    Bx��  �          @O\)>��
@L(����33B�ff>��
@9����33��ffB�G�                                    Bx���  �          @b�\>�33@[��!G��%p�B�W
>�33@E��У���33B�
=                                    Bx��N  �          @qG�>�z�@e���33��ffB��
>�z�@Dz��
�H�p�B�W
                                    Bx���  �          @j=q>���@W
=��33���B���>���@2�\����RB�\)                                    Bx���  �          @j=q=��
@W������33B�ff=��
@333����
B��H                                    Bx��@  �          @l(���@e�>\)@�
B���@]p��k��o�B���                                    Bx���  �          @g���ff@S33?���A�=qBî��ff@c�
>�{@�{B=                                    Bx��  �          @e��.{@Y��?Y��A\z�Bʙ��.{@`  �B�\�FffB���                                    Bx�2  �          @Tzῴz�@=p�>\@�p�B�
=��z�@<(���\�p�B�Q�                                    Bx�)�  �          @p���.{@$z���33C�{�.{@�H�c�
�[
=C
W
                                    Bx�8~  �          @hQ��
�H@4z�?.{A,��B�G��
�H@8�þ8Q��5B���                                    Bx�G$  �          @_\)��\)@C33?W
=A`��B�ff��\)@J�H���Ϳ�=qB��f                                    Bx�U�  �          @i�����\@Z�H?�A�
B�#׿��\@Z�H��\���B�#�                                    Bx�dp  �          @`  ��p�@N�R?#�
A(��Bݙ���p�@QG�������(�B�\                                    Bx�s  �          @hQ쿋�@[�>L��@Mp�B�G����@U�O\)�O\)B�.                                    Bx���  �          @n�R��  @^�R>aG�@\��B۞���  @Y���L���G�
B܊=                                    Bx��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��  �          @p  ��@9��?��
A�=qB�\)��@E�>��@\)B�.                                   Bx���  �          @o\)��@#33?��A�\)C�f��@4z�?�A   C�                                   Bx��T  �          @p�׿�@C33?�Q�A�z�B�(���@P��>��@�  B�                                    Bx���  �          @l(���ff@J=q?��A�ffB�B���ff@U=�G�?��B��H                                    Bx�٠  �          @j=q���@N�R?��
A��
B�����@X��=u?p��B�.                                    Bx��F  �          @e��(�@P��?J=qAL��B����(�@Vff�8Q��8��B��                                    Bx���  �          @c�
�.{@3�
@Q�Bp�B��.{@S�
?�p�A��B�                                    Bx��  �          @e��=q@QG�?�=qA�Q�B�p���=q@`��>���@�B���                                    Bx�8  �          @b�\>�@`��>Ǯ@ʏ\B���>�@^{�!G��$��B�                                    Bx�"�  �          @j�H�333@c33>�ff@�=qB�k��333@a녿z���Bʊ=                                    Bx�1�  �          @n{��ff@aG�>��@��B�(���ff@`  �(���B�k�                                    Bx�@*  �          @xQ쿠  @j�H>�?��Bي=��  @c�
�k��\(�Bڮ                                    Bx�N�  �          @z�H�\@`  ��ff���B��f�\@N{�����
B�\)                                    Bx�]v  �          @o\)��G�@N�R�+��$��B����G�@9����ff���HB�G�                                    Bx�l  �          @q녿��R@X�ÿB�\�9�B�aH���R@A녿�
=��  B�.                                    Bx�z�  �          @p�׿��@R�\�u�q��B��ÿ��@8Q�������\B��                                    Bx��h  �          @y����ff@XQ쿵��33B�#׿�ff@5�z���B�=q                                    Bx��  �          @u���z�@S33��(���(�B�33��z�@4z��ff��\B�                                      Bx���  �          @u���p�@J=q��  ���HBޏ\��p�@"�\�$z��)ffB�R                                    Bx��Z  �          @r�\����@I����33���
B�LͿ���@#�
�p��"(�B�Q�                                    Bx��               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Ҧ  �          @@  ����?��
��p��%��CW
����?+�����C�HC                                      Bx��L  �          @i���ff?��
=�!��C@ �ff?�=q�1G��F=qC��                                   Bx���  �          @u��Q�@1G����R��
=B�(��Q�@  �p���HC��                                    Bx���  �          @u���(�@P  >�p�@��
B�k���(�@N{�
=q�33B���                                    Bx�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��  �          @n�R�33@E���
���RB�33�33@8Q쿔z����B���                                   Bx�*�  �          @xQ��
�H@L��<#�
>\)B�p��
�H@E��aG��Qp�B��                                   Bx�90  �          @w���p�@Tz�=���?���B񞸿�p�@N{�Q��D��B�33                                    Bx�G�  �          @w���@S�
?�@��B�33��@Tz�����=qB���                                    Bx�V|  �          @w
=�33@Mp�?!G�AG�B�=q�33@P�׾�\)��B�u�                                    Bx�e"  �          @s�
�
=@8��?(�A(�C5��
=@<(��W
=�J�HC ��                                    Bx�s�  �          @I����Q�?�
=�k���\)B��ͽ�Q�?�\)�����2=qB�#�                                    Bx��n  �          @e�?��H?�33�0  �I��A�=q?��H>�33�=p��`  A!                                    Bx��  �          @b�\?��
?����%��<\)BI?��
?����?\)�h  B\)                                    Bx���  �          @a�>aG�@X�ÿaG��g�
B��=>aG�@AG���\��Q�B���                                    Bx��`  �          @a�>�\)@X�ÿh���p(�B���>�\)@AG���ff��(�B��                                     Bx��  �          @c33�B�\@QG�������p�B��׾B�\@2�\�
=q�B�Ǯ                                    Bx�ˬ  T          @e�aG�@C�
��G�����B�
=�aG�@{�!��633B���                                    Bx��R  �          @h��>�ff@`  �z��{B�� >�ff@Mp���G���B�33                                    Bx���  �          @l(�>�33@`�׿h���h  B���>�33@HQ��=q���HB�8R                                    Bx���  �          @e�#�
@Z�H�B�\�G�
B�B��#�
@E��z���\)B�aH                                    Bx�D  �          @e�>���@^{�����HB�>���@Mp���33��
=B��R                                    Bx��  �          @i��>W
=@b�\���{B�L�>W
=@P  ��  ��B��R                                    Bx�#�  �          @j=q>�\)@c33�W
=�T��B��H>�\)@L(���G����B��)                                    Bx�26  �          @h��>��@W
=��������B�L�>��@8Q��
�H��B��)                                    Bx�@�  �          @XQ�?�ff@��?��A�B��R?�ff@1�?Q�Axz�B���                                    Bx�O�  �          @G�?fff@ff?���A��
B�  ?fff@'�?+�AZ=qB�{                                    Bx�^(  �          @p  �Q�@HQ������Q�B�z�Q�@!��%�0�B�
=                                    Bx�l�  �          @qG���Q�@(Q��)���3BÅ��Q�?�=q�P  �p�
B�.                                    Bx�{t  �          @b�\���@��z��,(�Bͅ���?�(��8Q��g�
B�k�                                    Bx��  T          @k��B�\@9���
=�\)B�k��B�\@\)�3�
�Gz�Bٞ�                                    Bx���  �          @g
=�Tz�@0������ffB�Q�Tz�@ff�333�L��B�\                                    Bx��f  �          @e�W
=@ �����*33B�
=�W
=?�\�@  �c\)B��                                    Bx��  �          @e���  @z��!��4\)B➸��  ?����C�
�j�
B�{                                    Bx�Ĳ  �          @c33��Q�@G��(���A�\B�(���Q�?�  �E��r�
C��                                    Bx��X  �          @>�R��Q�?�z��z��M�RCB���Q�?@  �'
=�vffC�f                                    Bx���  �          @{��
=?��\���K�CͿ�
=>����lp�Cٚ                                    Bx��  �          @AG����H?}p��(��X�RC�����H>��
�'��qffC'��                                    Bx��J  �          @J�H��  ?�\)�%��X��CLͿ�  ?&ff�6ff�~��C��                                    Bx��  �          @W
=����?����,(��W��C	�Ό��?��<(��x\)C�H                                    Bx��  �          @:�H�   ?���9z�B�k��   ?���!G��s\)B�(�                                    Bx�+<  �          @Mp�>��@
=�	���)ffB���>��?��H�,���e��B��f                                    Bx�9�  �          @S�
�W
=@���(��)�RB��f�W
=?�p��/\)�e�B���                                    Bx�H�  �          @]p����
@8Q��{��B��
���
@33�"�\�>z�B�\                                    Bx�W.  �          @W
=����@3�
��\� �\B�B�����@  �(��<p�B�B�                                    Bx�e�  �          @[�����@E��p����B�8R����@&ff�\)�!B                                     Bx�tz  �          @\(��\@A녿�=q����B�k��\@ ����
�(��B�Q�                                    Bx��   �          @`�׿xQ�@(Q���
=Bܔ{�xQ�@   �,���J\)B��f                                    Bx���  �          @Z�H�aG�@'
=� ����B�{�aG�@   �'��H�B�B�                                    Bx��l  �          @`  ��33@'��   �(�B�k���33@ ���'
=�AB�                                     Bx��  �          @Z�H��G�@-p���ff��G�B��H��G�@
=q�(��6�
B�=q                                    Bx���  �          @]p��&ff@Dzΰ�
����B˸R�&ff@(������B�u�                                    Bx��^  �          @^�R��\)@Z=q�(��"�HB�aH��\)@HQ쿼(���=qB�33                                    Bx��  �          @X��>�(�@QG��+��8Q�B�.>�(�@?\)���R��{B���                                    Bx��  �          @X��>�@N�R�^�R�p  B���>�@9������B���                                    Bx��P  �          @[�>�@Vff������HB�z�>�@G������=qB�\)                                    Bx��  �          @\(�>���@X�ý#�
�5B�� >���@QG��h���v�RB�                                      Bx��  �          @W
=?�@S33�L�ͿfffB�L�?�@J�H�c�
�w\)B��\                                    Bx�$B  �          @Q�?�R@G
=�!G��3�
B�Q�?�R@6ff��33�ͅB�G�                                    Bx�2�  �          @`  ?0��@J�H?��RA���B�k�?0��@X��>���@У�B��                                    Bx�A�  �          @R�\>8Q�@N�R<��
>�(�B���>8Q�@HQ�L���dQ�B�p�                                    Bx�P4  �          @P  ��@J�H�aG��vffB�\��@@�׿�G����RB�
=                                    Bx�^�  �          @N�R=�Q�@G
=�W
=�x��B��
=�Q�@<�Ϳ}p����\B��                                    Bx�m�  �          @Q�?�p�@4z�u����By�?�p�@*�H�p�����Bt=q                                    Bx�|&  �          @P��?�{@<��=���?�ffB���?�{@8Q�&ff�7�
B��{                                    Bx���  �          @N{>.{@J=q?   A33B��q>.{@K����
���B���                                    Bx��r  �          @QG�>k�@L(���ff� ��B��R>k�@>{���R���HB��                                    Bx��  �          @Tz�?\)@Fff�k���(�B�z�?\)@1G������RB��                                    Bx���  T          @Vff>�
=@K��Y���m��B�{>�
=@7���\)��=qB�u�                                    Bx��d  �          @S33?�{@=p�>�{@���B��?�{@<�;����RB��=                                    Bx��
  �          @Mp�?���@8��>�=q@��B��
?���@7������B�z�                                    Bx��  
}          @L(�?L��@E�=��
?�G�B�?L��@@  �0���Hz�B�L�                                    Bx��V  �          @HQ�?Y��@@  ���Q�B�=q?Y��@7��\(���ffB��                                    Bx���  
�          @HQ�?��@;��k���
=B��)?��@1녿u���\B��                                    Bx��  
�          @Fff?��@333�(��6�\B�aH?��@#�
�����ffB�Ǯ                                    Bx�H  �          @Z=q?�{@6ff��(����B�Ǯ?�{@�Ϳ����Bs��                                    Bx�+�  �          @a�?Ǯ@2�\��(��ŅBsp�?Ǯ@���{Baff                                    Bx�:�  
�          @\��?��H@<�Ϳn{��=qB�=q?��H@'��У����
B�\)                                    Bx�I:  T          @l(�?5@g
=�\)���B��R?5@]p����
��ffB�Ǯ                                    Bx�W�  �          @k�?h��@`�׾�
=�ҏ\B�?h��@S33���
���B�=q                                    Bx�f�  
�          @h��?�=q@\(������p�B�8R?�=q@N�R���R���\B�#�                                    Bx�u,  
Z          @n{?.{@hQ쾽p���  B��
?.{@[���G����\B���                                    Bx���  T          @qG�>�p�@o\)��z���{B��R>�p�@c33���H��p�B��                                    Bx��x  
(          @xQ�u@qG����
��z�B��f�u@i���s33�iG�B���                                    Bx��  �          @~�R���
@h��?�(�A��HBӽq���
@tz�>�\)@�33B�B�                                    Bx���  "          @xQ�z�@k�?��Az�\B��Ϳz�@tz�=�G�?���B�#�                                    Bx��j  �          @k�?c�
@c�
�8Q��0  B���?c�
@Y����ff��z�B�Ǯ                                    Bx��  "          @l��@�@,�Ϳ������BO�@�@G�� ���B<Q�                                    Bx�۶  "          @mp�@@������33B3�\@?�{�
=q��B                                    Bx��\  �          @mp�@9��>���
=q��R@8Q�@9���Ǯ�Q��\)C�/\                                    Bx��  
�          @e@.{?\)�  �{A:�H@.{�#�
�z��!�
C��                                    Bx��  
�          @l(�@'
=?333��R�*{Ap��@'
==�\)�$z��2ff?��                                    Bx�N  �          @g�@\)?���(���;�Aǅ@\)>�p��3�
�LffAQ�                                    Bx�$�  �          @i��@!G�?Y���   �,�
A�ff@!G�>W
=�(Q��8ff@���                                    Bx�3�  
�          @g�@�?���p��+��A�@�>���(���;��A��                                    Bx�B@  T          @n{@,(�>��   �*(�A"ff@,(��\)�#33�-�C�xR                                    Bx�P�  
�          @n�R@:�H>�p�����@�  @:�H�B�\�33��
C�
                                    Bx�_�  
�          @u�@(��    �.�R�7��=�\)@(�ÿ(���)���0�C��q                                    Bx�n2  
(          @xQ�@<(�<#�
�!G��"(�>.{@<(���������C�7
                                    Bx�|�  
�          @xQ�@G
=>����33�(�@�@G
=��=q�33�ffC��                                    Bx��~  �          @p  @9��>�����{@8Q�@9����
=�33�ffC��R                                    Bx��$  �          @b�\@@  >\��=q��p�@�@@  ���
��{���C�E                                    Bx���  T          @[�@*=q?=p��G����Aw�@*=q>k�������@�(�                                    Bx��p  "          @\��@=p�>������H��
=@��@=p�����p���Q�C���                                    Bx��  "          @\(�@!G�>\����&�AQ�@!G��8Q���\�(�
C���                                    Bx�Լ  2          @c33@'�>������%��@�=q@'���=q���&G�C��                                    Bx��b  
�          @c�
@!G�?&ff�(�� �RAg�@!G�=�����\�(��@�
                                    Bx��  �          @W
=?��H@�����R��Q�BZ
=?��H?�(��G��
=BD�                                    Bx� �  "          @\(�?��@
=������33BM��?��?�z����\)B5ff                                    Bx�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�,�  V          @c33?�=q@8�ÿ��\����Bu�\?�=q@   ��
=�p�Bg{                                   Bx�;F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�X�  �          @��\@*�H���Q����C��q@*�H�;���p����RC�0�                                    Bx�g8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��h  �          @~�R@c33���\��(���
=C�  @c33�����
=��Q�C��\                                    Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Z  �          @*=q?�\)?��\�-p�B33?�\)?���&ff����B=q                                    Bx�   "          @��?B�\@p���Q��Q�B��?B�\@Q�(��p��B��                                    Bx�%�  �          @%?�=q?�p�>���@�RB-\)?�=q?�G���Q��   B/=q                                    Bx�4L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Q�  �          @.�R@G�?�p���z���ffBG�@G�?�\)�!G��W�A�ff                                    Bx�`>  T          @.{@��?�  �#�
�aG�Aܣ�@��?��H��{��(�A��
                                   Bx�n�  �          @.�R@33?��H>�@2�\B�@33?����aG�����B ��                                   Bx�}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��"  �          @C33?���@?�33A�  Brff?���@'
=?O\)Aw�B|�\                                   Bx���  T          @J=q?�{@�?�ffA�BV��?�{@�R?}p�A�{Bdz�                                   Bx��n  "          @X��?�(�@+�?�=qA�B��?�(�@>�R?h��A{�B�z�                                    Bx��  �          @g
=?�\)@8��?�\)A�
=B���?�\)@O\)?��A���B���                                    Bx��  T          @j�H?xQ�@H��?��AӮB���?xQ�@[�?Y��AV�HB�u�                                    Bx�`  �          @l(�?���@G�?�\)A�p�B�  ?���@Z=q?W
=ATz�B�\                                   Bx�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�-R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�YD              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�v�  \          @�  ?���@@  ?��HA�BkQ�?���@XQ�?���A�p�Bv�                                   Bx��6  "          @P��@   ?��
?�\)B�\B&@   @
�H?�33A��B==q                                   Bx���  
�          @o\)@@Q�?��HA��BC=q@@1G�?���A��
BTG�                                    Bx���  
�          @\)?�=q@;�?�p�A�RBh{?�=q@Tz�?��RA�p�BtQ�                                    Bx��(  "          @}p�?�G�@:=q@�\A�  Bk��?�G�@S�
?��A��Bx(�                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��t   T          @��\?�=q@^{?�A��
B�?�=q@qG�?Q�A9�B�\                                    Bx��  �          @��?���@N�R?���AٮB~�
?���@dz�?�ffAo�
B��H                                    Bx���  �          @�  ?���@>�R?�
=A��HBj(�?���@Vff?�
=A�\)Bu�                                    Bx��f  �          @q�?�  @B�\?�\)A��
B~��?�  @U?\(�ATz�B��=                                    Bx�	              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��   T          @j�H?(�@^�R?}p�A{�B�{?(�@g
=>#�
@��B���                                    Bx�&X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�4�   T          @`  =u@Z�H?=p�AB�\B�
==u@`  ���
��ffB�{                                    Bx�C�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�RJ   T          @z�H>#�
@vff?5A(  B��>#�
@z=q�L���<(�B�.                                    Bx�`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�o�   T          @�=q<#�
@�녾L���2�\B���<#�
@z=q������B���                                    Bx�~<  �          @�  ���
@~{>�=q@tz�B�uü��
@{��+��{B�z�                                   Bx���  
�          @���=���@�  >�z�@�B�(�=���@}p��#�
�(�B��                                   Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��z   T          @���#�
@��<�>�p�B���#�
@�  �n{�Q�B�#�                                    Bx��               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   T          @����=q@��\�\��G�B��3��=q@xQ쿨������B��                                    Bx��l  �          @��>.{@�����������B��>.{@w
=��p�����B��f                                    Bx�  �          @���=��
@�Q����Q�B���=��
@qG���Q����B�u�                                    Bx��  �          @���=�\)@~{�!G��  B��=�\)@n{�����p�B���                                    Bx�^  �          @��׾k�@y���p���X��B��3�k�@dz�����׮B�L�                                    Bx�.  �          @�Q쾔z�@u�����y�B��\��z�@_\)��Q���B�k�                                    Bx�<�  �          @\)����@w
=�n{�YB��
����@b�\����Q�B��                                    Bx�KP  �          @�  >\)@w
=��G��j�\B�>\)@aG�������B�\)                                    Bx�Y�  �          @�Q�=���@x�ÿp���X��B�
==���@dz�������
B�Ǯ                                    Bx�h�  �          @�  �#�
@tzΐ33����B�=q�#�
@\���   ���B�B�                                    Bx�wB  �          @~{>Ǯ@s�
�fff�S�B�\)>Ǯ@`  ��G��ԸRB�W
                                   Bx���  �          @}p�?��@s33���
���B��?��@g�������z�B���                                   Bx���  �          @{�?�  @l�;�{��  B��3?�  @aG���Q�����B��                                    Bx��4  �          @|(�?�p�@i���G��8Q�B��R?�p�@W
=��{����B��
                                    Bx���  �          @|��?��\@r�\���
����B��R?��\@k��fff�T(�B��)                                    Bx���  �          @}p�@G�@W��������Bl=q@G�@K�����z�Bfp�                                    Bx��&  �          @{�?���@]p���G���(�B��=?���@QG����R��=qB�
                                   Bx���  �          @|(�?�{@e�:�H�+�B�k�?�{@Tz������B�aH                                   Bx��r  �          @z�H?��
@b�\�   ��{B�B�?��
@U������B���                                    Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�	�   T          @hQ�@p�@$z�?=p�A;�
B8�@p�@*�H=�?�B=\)                                    Bx�d  �          @\)@�H@J�H>�  @hQ�BR\)@�H@H�þ��H���BQ\)                                    Bx�'
  �          @~�R@p�@S�
�#�
�8Q�B`�@p�@N{�G��5G�B]�H                                    Bx�5�  �          @~�R@33@W
=��  �k�Bjz�@33@Mp����
�qBe�R                                    Bx�DV  �          @���?333@fff��=q���\B�  ?333@HQ����  B���                                    Bx�R�  �          @���?��R@aG���Q����B�L�?��R@E�{�  B�\)                                    Bx�a�  T          @�=q?G�@`�׿�\)��(�B�
=?G�@>�R�(���"z�B��q                                    Bx�pH  �          @���?W
=@W�������B�(�?W
=@333�333�/ffB��R                                   Bx�~�  �          @��?!G�@fff��p��ʸRB�B�?!G�@E�!G��Q�B�#�                                   Bx���  �          @\)?�  @c�
��p����
B���?�  @G��G��
�\B�u�                                    Bx��:  �          @�  ?+�@`�׿�\��p�B��?+�@@  �"�\��\B���                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���   T          @�  ?��
@Z=q����(�B��3?��
@?\)�
�H�B{�                                    Bx��,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��x  
�          @x��@�R@E��0���$��BXp�@�R@5������RBOG�                                    Bx��  �          @|��@�@@�׿�����BSff@�@*=q����{BE��                                    Bx��  T          @|��@�\@:�H��=q��{BO�@�\@!녿�p�����B?=q                                   Bx�j  "          @z�H@ ��@7
=�p���^ffBC
=@ ��@#�
��=q���RB6ff                                   Bx�   �          @|(�@!�@2�\��
=��p�B?=q@!�@���ff��G�B/                                    Bx�.�  "          @|��@!�@1G����R��ffB>\)@!�@���������B.�                                    Bx�=\  �          @\)@'�@0  ��(����\B9��@'�@�ÿ�=q�ڏ\B)z�                                    Bx�L  �          @���@'
=@4zῙ����
=B<�H@'
=@p���=q��=qB-=q                                    Bx�Z�  
�          @�G�@!G�@333����G�B@33@!G�@����\��33B.{                                    Bx�iN  
�          @�G�@(��@�����G�B�H@(��?Ǯ�%�� =qA�{                                    Bx�w�  "          @�=q@�R@33�  �Q�B+G�@�R?��H�.�R�(ffB
33                                    Bx���  �          @�G�@#33@33�
=��z�B(G�@#33?޸R�%���B	ff                                    Bx��@  T          @�G�@��@��������B6ff@��?���)���$(�BQ�                                    Bx���  �          @���@��@ff�\)��HB1��@��?�G��.�R�*33B��                                    Bx���  �          @���@�@	���   �=qB-33@�?�  �;��<(�Bz�                                    Bx��2  
�          @�=q@ ��@��Q���
B �H@ ��?�  �3�
�/�A�33                                    Bx���  �          @��\@\)@p���
��  B2��@\)?�z��%����B�                                    Bx��~  	�          @��?�{@l(�������B��?�{@S�
��p���p�B�ff                                   Bx��$  
�          @�=q?У�@dz῁G��g
=B���?У�@N�R�������
B}
=                                   Bx���  
�          @��@	��@G����
��ffB]��@	��@*�H�{�  BL                                    Bx� 
p  
�          @�z�@p�@(���   ��\)B<
=@p�@ff�#�
��B!�                                    Bx�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx� '�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx� 6b  \          @���@�@?\)������=qBR�\@�@%�G�����BBG�                                    Bx� E  �          @�=q@��@Fff��=q�v�RBQ�@��@0�׿��
��{BC��                                    Bx� S�  T          @��@ ��@>{��ff��{BG{@ ��@%���(���G�B6��                                   Bx� bT  "          @�G�@#�
@0  �������B<�@#�
@���R��
=B)�
                                   Bx� p�  T          @~{@(��@�Ϳ����
Bz�@(��?�Q�����p�BG�                                    Bx� �  
�          @z=q@33@QG�?\)Az�Bg��@33@S�
�k��W�Bh�
                                    Bx� �F  �          @r�\@�R@B�\�u�i��BW=q@�R@9���u�jffBQ�                                   Bx� ��  
�          @p  @�H@4z�������BEff@�H@)���������B>{                                   Bx� ��  "          @vff@(Q�@.{�(���=qB7@(Q�@�R������\B-p�                                    Bx� �8  
�          @vff@�H@9���+��!BH�H@�H@*=q�����  B>��                                    Bx� ��  �          @xQ�@1G�@)��>��@�p�B.�@1G�@+��k��X��B/�                                    Bx� ׄ  
�          @|(�@5�@#33=���?�(�B(=q@5�@   ��\��
=B%��                                    Bx� �*  
�          @��@3�
@4z�G��2=qB4ff@3�
@#33��
=��p�B(��                                    Bx� ��  �          @���@5�@B�\��G��[
=B<{@5�@-p����H��=qB.��                                    Bx�!v  "          @�
=@.�R@=p���G����RB==q@.�R@$z����p�B-=q                                    Bx�!  T          @��R@5�@;�����i�B7�@5�@%��p����B)�H                                    Bx�! �  �          @��@L(�@#33�
=q���BG�@L(�@�����
=B�                                    Bx�!/h  �          @�
=@>�R@(Q쿷
=��  B%@>�R@p�� �����Bff                                    Bx�!>  �          @�ff@9��@(Q������B(�H@9��@(��Q���\B
=                                    Bx�!L�  "          @���@\)@p�����B2�\@\)?�=q�7
=�*�
B(�                                    Bx�![Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  