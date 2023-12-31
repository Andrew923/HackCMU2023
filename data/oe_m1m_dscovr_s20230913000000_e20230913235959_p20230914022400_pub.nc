CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230913000000_e20230913235959_p20230914022400_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-09-14T02:24:00.825Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-09-13T00:00:00.000Z   time_coverage_end         2023-09-13T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��f�  �          A,z�������
�p  ���HC`�{������\��\)� (�Cc��                                   Bx��uf  �          A2=q���ə��^�R��=qCX�����޸R������HC[p�                                   Bx���  T          A-�������˅�(�CDp����{������
=CM��                                    Bx����  �          A'
=����
=�������CA������XQ���
=�ٙ�CI�H                                    Bx���X  "          A���ڏ\�8Q�����0�\C9���ڏ\������$
=CF�3                                    Bx����  �          A(��У׿J=q����,�RC:�f�У���\������CG^�                                    Bx����  "          A&ff���ÿ�Q����1�
C=O\�����<����
=�"{CJ{                                    Bx���J  �          A7\)� Q�}p����3�HC;� Q��=p���=q�&�CHE                                    Bx����  T          AC�
��?����  �)z�C,�����z�H�(��)�HC:=q                                    Bx���  
�          AF=q�33@p��(��'\)C&!H�33=�Q�����.�C3s3                                    Bx���<  �          AG�
�{?Ǯ�	��.�C*\�{�#�
��
�1�HC8#�                                    Bx���  �          AIG��Q�?������3
=C)k��Q��R���6ffC8�                                    Bx���  "          AH���@Z�H�ff�"z�C��?��\��
�1G�C+��                                    Bx��%.  �          AE���	�@�����z��{C+��	�@{�z��.�C#�                                    Bx��3�  �          ADz��@������  C�)�@A��\)�.=qC �                                    Bx��Bz  
�          AC33�
=@�z���\��RC���
=@9�����,
=C!                                    Bx��Q   �          AB�R�@�Q��������C�@O\)���0��C8R                                    Bx��_�  	�          AE��\)@�����\)��C���\)@n�R�
=�2z�C�                                    Bx��nl  +          AD(���z�@�����{C�3��z�@��R����,{C��                                    Bx��}            AA�����R@�ff�ָR��C)���R@������(Cs3                                    Bx����  �          A?
=����@����H�Q�C� ����@�=q���&��C��                                    Bx���^  
�          A@  ��@�(���ff��C5���@�Q�����)�HC�                                     Bx���  T          A?\)��
=@�{��
=�
=C���
=@���33�-=qC
�                                    Bx����  T          AB=q��(�@�p���
=� ��C�\��(�@�����&p�CE                                    Bx���P  �          AA�����@�=q��
=��{C����@�  ���$=qC޸                                    Bx����  �          AA���(�A  ��p���C Ǯ��(�@Ϯ��=q� �C�H                                    Bx���  +          AE�����A
ff�ȣ����RB��H����@أ��(��(G�C@                                     Bx���B            AL����=qA\)���R���\B�����=q@��\��{�z�C�                                    Bx�� �  "          AF�\��A�H��\)��33B���A���33��B�                                      Bx���  T          AHQ���=qA$Q���p���B�Ǯ��=qA  ��(��\)B��                                    Bx��4  
�          AG���\)A+����
����B���\)A33��ff���B�\)                                    Bx��,�            A?
=�HQ�A,���\����ffB�Q��HQ�A���\��\)B�=q                                    Bx��;�  T          AG
=���A   ��z���p�Cz����@˅��  ���C�                                    Bx��J&  "          AE���z�A�R�����C5���z�@����p���  C	p�                                    Bx��X�  |          A1��1G�A+\)��(���G�B�  �1G�A ���z�H���HB��
                                    Bx��gr  �          A8Q���  A((��������HB�ff��  A{�n{��\)B���                                    Bx��v  �          A7����
A5>�33?�\B�\)���
A1��$z��P(�B���                                    Bx����  �          A8Q��\)A2�R����C�
B����\)A*=q�\�����B��f                                    Bx���d  �          A:{�*=qA2�R��z���Q�B��*=qA'
=�����BД{                                    Bx���
  �          A9G����RA6ff������B��
���RA,  �|(����BøR                                    Bx����  �          A7���33A4Q쿇����\BĞ���33A)�{����RBŞ�                                    Bx���V  h          A4  ��ffA/\)<��
=�G�B��{��ffA)���3�
�k�B�
=                                    Bx����  "          A3����A1����\)��{B�k����A+33�;��s33B��                                    Bx��ܢ  
�          A8  @
�HA1G����"�HB���@
�HA"�H��=q���HB��                                    Bx���H  �          A7�
@*�HA2=q�k���33B���@*�HA(Q��qG���G�B�u�                                    Bx����  
�          A333?���A-��\��B��?���A z�������G�B�#�                                    Bx���  
c          A5�?^�RA0  �
=�AG�B��?^�RA (���z��؏\B�\                                    Bx��:  A          A6{?���A3�
���:=qB�?���A+33�^{��
=B�(�                                    Bx��%�  T          A4(�>�  A3\)��R�K�B�z�>�  A*�\�aG���  B�\)                                    Bx��4�            A4  ���A/������!G�B��H���A!p������G�B�L�                                    Bx��C,  
�          A2{�0��A/33��p��
=B�=q�0��A!�������(�B��
                                    Bx��Q�  
�          A1p���p�A0  ��{��Q�B��)��p�A%G��|(����B��                                    Bx��`x  
�          A-G���{A'\)��z��\)B��)��{A�\���H���B�{                                    Bx��o  T          A*�\�1G�A�\�*=q�h  B�8R�1G�A���p���\B֮                                    Bx��}�  �          A(����A Q�(���k�B�8R��A�
�QG����BШ�                                    Bx���j  
�          A*{��HA"ff?\)@C�
B�Ǯ��HA
=�Q��;�
B�Q�                                    Bx���  
Z          A'
=���RA�>���?�33B��
���RA��p��DQ�B�                                      Bx����  
Z          A+33�+�A���=q�U�Bҙ��+�A��������Q�B���                                    Bx���\  
�          A)��N�RA�c33���B���N�RAp���z��Q�Bߊ=                                    Bx���  "          A$���Z=qA(��:�H��B�ff�Z=qA�\��Q����B�Q�                                    Bx��ը  
c          A��3�
A=q��R�n�\B�\�3�
@�p������B�
=                                    Bx���N  �          @�?O\)��=q?�{A��C��H?O\)��(�?ǮB+z�C�\)                                    Bx����  T          @-p�?�Q�^�R@ ��BH=qC�p�?�Q쾞�R@
�HB_G�C��                                    Bx���  �          @K�?��<#�
@5B�G�>�(�?��?B�\@.�RBr  A�\                                    Bx��@  �          @l(�?���ff@
=B@ffC�8R?��\@#�
BVz�C�,�                                    Bx���  T          @���@G
=��z�@Dz�B'�HC��3@G
=�
=q@UB:G�C�{                                    Bx��-�  �          @���@P�׿E�@^�RB8�C�U�@P��>8Q�@c�
B=�@L(�                                    Bx��<2  �          @�ff@+�@�(�@���B=qB~(�@+�@�  @(�A��
B��                                    Bx��J�  �          A	�@H��@ۅ@��A�\B��
@H��@�{@�Aa�B���                                    Bx��Y~  �          A�@0  @��H@Z�HA���B�L�@0  A�?�Q�@�(�B���                                    Bx��h$  �          A��@�R@�{@C�
A�
=B���@�RA�?Q�@��B��=                                    Bx��v�  T          A�@B�\@�\)@�(�B�RB�\)@B�\@���@8��A�z�B�\)                                    Bx���p  ]          AG�@Fff@ҏ\@�z�BQ�B��{@Fff@���@[�A�  B���                                    Bx���  
�          AG�@E�@�
=@�p�BG�B�\)@E�A ��@7
=A��HB��                                    Bx����  T          Ap�@S�
@�p�@�=qB z�B��@S�
@�
=@1G�A�z�B��                                    Bx���b  �          A��@j=q@��
@�ffA�\)B{  @j=qA�@&ffA�=qB�k�                                    Bx���  "          A@e@�33@��A�(�B|�R@eA z�@�AmG�B��
                                    Bx��ή  
�          A�H@5@��@���A���B�aH@5A��@)��A��\B�G�                                    Bx���T  
�          A\)@"�\@�G�@���A�(�B�@"�\A�
@Al(�B��                                    Bx����  �          A  @��@�@�33A�
=B�B�@��A  ?�{A?
=B�G�                                    Bx����  T          A@��
@��@�=qB 33BTp�@��
@�@=p�A�=qBe                                    Bx��	F  
�          A��@���@��H@��A��BS��@���@�p�@<��A��
Bd��                                    Bx���  �          A  @s�
@��H@�\)A�  Bv��@s�
A�H@7
=A��\B�
=                                    Bx��&�  T          A�
@+�@�=q@��B ��B�  @+�A
�R@333A�\)B���                                    Bx��58  +          A�@�
@���@���A�
=B��f@�
A��@%�A}�B���                                    Bx��C�  �          A�@(�A  @�(�A�{B�G�@(�A33@��AQ�B�=q                                    Bx��R�  �          A��@
=qA�\@��HA�ffB�W
@
=qA��@
�HAQG�B�Q�                                    Bx��a*  ]          A�׿�z�A	��@4z�A��Bƣ׿�z�A��>�33@(�BŽq                                    Bx��o�  
�          A�\���Aff@7
=A�  BЊ=���A>�(�@.{B��                                    Bx��~v  
�          A=q��G�A{@QG�A�  B�k���G�A33?Tz�@�
=B��                                    Bx�  T          Az�.{A��@~{AʸRB�LͿ.{A�?�  Ap�B��                                    Bx��  
�          A�þk�A\)@�\)A�  B�.�k�A��?��A2�HB��f                                    Bx�ªh  
Z          A�=�Q�A@��\AծB���=�Q�A�H?�A)B��f                                    Bx�¹  T          A�H��G�A	G�@Mp�A�  B�uý�G�A{?333@�(�B�aH                                    Bx��Ǵ  
�          A=q>L��Az�@(�Ax(�B�\)>L��A���Ϳ#�
B�u�                                    Bx���Z  �          A  ?�A
{@K�A��B�  ?�A�R?&ff@�=qB�k�                                    Bx���   �          Ap���A	@dz�A��B��{��A(�?��\@�G�B�u�                                    Bx���  �          A{�5AG�@A�A��B�33�5AG�>�G�@-p�B��R                                    Bx��L  
�          A  ��G�A�H@Dz�A�z�B����G�A�?(��@�=qB��q                                    Bx���  �          A(����
@�(�@A�A�Q�B�����
A�R?333@�Q�B�\                                    Bx���  
�          A
=q��A z�@Dz�A�\)B�k���A	�?0��@��B�aH                                    Bx��.>  
�          A  ���
@��
@EA��\B��q���
A�H?Q�@��B�k�                                    Bx��<�  ]          A�R���
@��@E�A�G�B�B����
A�?Q�@���B�=q                                    Bx��K�  
�          @�z�@
=q@�(�@l��A�B��f@
=q@��?�
=AD��B�u�                                    Bx��Z0  
�          A�H@�@���@o\)A�p�B�{@�@�p�?��A8��B��                                    Bx��h�  �          A
=?E�AQ�@N{A�p�B�aH?E�Ap�?=p�@�G�B�\                                    Bx��w|  �          Ap�>�{A	��@�
AW\)B�� >�{AG����@��B���                                    Bx�Æ"  
�          @�Q�@��H@��R@r�\A�  BI�
@��H@��\@{A���B[�                                    Bx�Ô�  
(          @�z�?�@�(�?�ffA��B��=?�@���O\)��p�B��{                                    Bx�ãn  �          @љ�?.{@�33>��?���B�Ǯ?.{@�(����q�B�W
                                    Bx�ò  
�          @�=q@@  @%�?�G�A�(�B#  @@  @>�R?s33AK33B3G�                                    Bx����  �          @�  @l(��n�R@5�A�z�C�XR@l(��1G�@q�B33C���                                    Bx���`  �          @�G�@�=q�/\)@XQ�A��C�g�@�=q��33@���B�C��                                    Bx���  
Z          @�Q�@~{�|��@�A�\)C��=@~{�E@]p�B
{C��                                    Bx���  �          @�ff@�  �a�@>{A�\)C���@�  �!�@w
=B
=C�Z�                                    Bx���R  �          @��@�{����@>{B�\C���@�{��G�@O\)BC�1�                                    Bx��	�  
(          @xQ�@%�����@��B��C��)@%��333@(�B)��C�j=                                    Bx���  
�          @hQ�@  �&ff>�Q�@��RC�ff@  ���?���A�C��H                                    Bx��'D  T          @���@Mp���녾��R�S33C�U�@Mp���
=?s33A"�HC���                                    Bx��5�  
Z          @�ff@������>�@���C��{@���o\)?У�Ay�C�
                                    Bx��D�  
Z          @�Q�@�Q��J�H?\(�A�
C��)@�Q��1�?޸RA��C�}q                                    Bx��S6  T          @�
=@���^{?�(�A[33C�Z�@���9��@=qA��C���                                    Bx��a�  "          @�Q�@\)��  ?���AP��C��{@\)�k�@!�A�=qC���                                    Bx��p�  �          @���@�  �i��?��Au�C��f@�  �C33@"�\A�G�C�.                                    Bx��(  T          @�p�@��H�z�@�HA�G�C���@��H��G�@?\)A�C���                                    Bx�č�  T          @���@�(����?�33A�p�C��q@�(����R@(�A�G�C��                                    Bx�Ĝt  T          @���@��H�%��=p���\)C�\)@��H�+�=L��?
=qC��                                     Bx�ī  T          @��\@�ff�Ǯ>8Q�?�33C���@�ff��
=?!G�@���C�K�                                    Bx�Ĺ�  �          @�p�@�p���=q?�{AhQ�C�k�@�p����\@z�A��C���                                    Bx���f  T          @˅@��
�Ǯ?���A�G�C��3@��
�s33@��A�{C�ff                                    Bx���  T          @�@�=q�+�?�@�p�C�k�@�=q��?5@��C�j=                                    Bx���  �          @�  @�{>��R�����\@c�
@�{>#�
����z�?���                                    Bx���X  
�          @�G�@�  ?&ff?�(�Ay�@�\@�  ?��?��HALz�A6�\                                    Bx���  
�          @�p�@�p�?�?�Q�A���@�=q@�p�?�=q?ٙ�A��A\                                      Bx���  �          @��@��\>Ǯ@Q�A��@��@��\?�=q@
=qAΣ�A`��                                    Bx�� J  T          @�Q�@�Q���@��A��
C���@�Q�>�=q@�A�G�@u                                    Bx��.�  ]          @�ff@��>�z�?�
=A���@��\@��?\(�?�  A���A<Q�                                    Bx��=�  }          @��@�33?��?�33A���AU�@�33?�{?�p�A�\)A�                                      Bx��L<  �          @�  @w
=@   @�
A�{A�
=@w
=@&ff?�{A�ffB�
                                    Bx��Z�  S          @l��?�G�@?\)=#�
?=p�B|?�G�@7
=�\(��l��Bx                                    Bx��i�  �          @��>�G�@w�>��@��B�(�>�G�@n�R���\�q�B��3                                    Bx��x.  
�          @�  ?���@�
=?��@�\)B�?���@�{�@  �B�                                    Bx�ņ�  ^          @��\@(�@���?^�RA
=B��{@(�@�녿(��љ�B��                                    Bx�ŕz  J          @��
@G�@��H?�z�A�33B��@G�@�z�>�=q@E�B���                                    Bx�Ť   "          @��
@33@���?uA.ffB�Q�@33@�������RB�
=                                    Bx�Ų�  
�          @�(�@ ��@�z�?J=qA�RB�=q@ ��@���&ff���B�p�                                    Bx���l  �          @�G�?�p�@��H?z�HA>�\B�p�?�p�@��\��=qB�G�                                    Bx���  �          @�
=?�{@��
?}p�A6�HB��?�{@�ff�����\B�#�                                    Bx��޸  �          @���?���@��\?p��A�B��?���@��H�c�
��B�(�                                    Bx���^  T          @�Q�?���@���?��
A33B��?���@�=q�G���Q�B�\                                    Bx���  �          @�=q?�  @��
?8Q�@�=qB��H?�  @�녿�{�)B���                                    Bx��
�  T          @�G�?�
=@�Q�?��@��RB�p�?�
=@�p������6�HB��                                    Bx��P  �          @θR?Ǯ@�ff?��@��HB��)?Ǯ@��H����@(�B�W
                                    Bx��'�  �          @У�?�ff@��@33A�z�B�{?�ff@ʏ\=���?h��B��\                                    Bx��6�  �          @��?���@�(�?�  A�p�B��=?���@�z�������B���                                    Bx��EB  �          @ƸR?W
=@�(�?�p�A�{B�?W
=@��
�.{�ǮB�ff                                    Bx��S�  �          @�p�=#�
@���@G�A�G�B�#�=#�
@�z�=�?��B�.                                    Bx��b�  �          @���>��
@���?�
=A5p�B�>��
@�
=�333��{B��
                                    Bx��q4  �          @�{?޸R@�
=?��\A8z�B�k�?޸R@�=q�&ff��33B��                                   Bx���  T          @��@N�R@��R@!G�A�  Bq�@N�R@ƸR?�R@�G�Bz                                     Bx�Ǝ�  
�          @��@i��@�ff@p�A�G�Bi�
@i��@�p�>�@o\)Bqz�                                    Bx�Ɲ&  
�          @���@e@�
=@�HA�G�Bk��@e@�p�>�G�@X��Bs(�                                    Bx�ƫ�  T          @�\@8Q�@�(�@\)A�  B�aH@8Q�@�Q�>�?��B���                                    Bx�ƺr  
�          @�\)@
=q@�  @:�HA���B�.@
=q@��H?=p�@��B�
=                                    Bx���  
�          @�ff@=q@׮@2�\A�=qB��@=q@��?�@}p�B�\)                                    Bx��׾  �          @�33?�@�\)@�\A�  B�W
?�@�=q=u?   B�#�                                    Bx���d  ^          @�G�@��@��@��A���B�B�@��@�(�>�@x��B�
=                                    Bx���
  
�          @�z�@AG�@�
=@C33A�Q�B|�@AG�@�(�?�G�@�B�                                    Bx���  "          @��
@P��@�@a�ABgQ�@P��@���?��HA_�
Bvp�                                    Bx��V  �          @�z�@g
=@���@�A�=qBg��@g
=@�z�>8Q�?�Q�BnQ�                                    Bx�� �  
�          @�\)@���@�Q�@�RA�33BR�@���@�  ?�@{�B[�\                                    Bx��/�  
�          Ap�@�  @���@Dz�A�Q�BJG�@�  @θR?���@�BV�                                    Bx��>H  T          A��@��H@���@^{A�{BHG�@��H@ҏ\?���A!p�BV��                                    Bx��L�  �          Ap�@�ff@���@�  A癚B;=q@�ff@ȣ�@
=Alz�BN�                                    Bx��[�  
Z          A=q@�p�@��@�A���BC��@�p�@Ϯ@\)A�(�BY
=                                    Bx��j:  �          AG�@�(�@�p�@�33BG�B@�\@�(�@��
@-p�A�BW�                                    Bx��x�  
�          A��@�Q�@���@���B  B(��@�Q�@��H@Dz�A�{BEp�                                    Bx�Ǉ�  �          A�@�{@���@��B33B3  @�{@��\@^{A���BR��                                    Bx�ǖ,  T          @���@l(�@��@��B��BT(�@l(�@�p�@:=qA�{Blp�                                    Bx�Ǥ�  �          @�R@U�@�33@@��AƏ\Bm(�@U�@���?�ffA�BxG�                                    Bx�ǳx  
�          @�\@Vff@��
@�p�B��Bh  @Vff@�@p�A��
By��                                    Bx���  
�          @��@W�@��@��\B  Bl{@W�@�G�@!G�A���B~z�                                    Bx����  
�          A\)@G
=@�z�@�(�B��B|�@G
=@�
=@	��At��B�p�                                    Bx���j  �          Aff@P��@љ�@aG�AͮB~  @P��@��H?���Az�B�                                    Bx���  T          A ��@k�@�p�?��HAD��Bx
=@k�@�33�#�
��G�Bzp�                                    Bx����  �          A�@E@�Q�@Z�HA��
B��
@E@�  ?��
@�B�8R                                    Bx��\  T          A��@���@��@z�A`Q�Bg@���@�녾�{�G�Bk�                                    Bx��  ^          A	�@���@ۅ@>{A�\)Bh\)@���@�ff?\)@n{Bp                                    Bx��(�            A	@fff@�
=@?\)A�ffB}�@fff@���>�ff@@��B�k�                                    Bx��7N  
�          Az�@&ff@�(�@w�A�{B���@&ffA (�?��AB�\                                    Bx��E�  �          A  ?���@�=q@���A�
=B��R?���Aff?�{ADz�B�Ǯ                                    Bx��T�  �          A��?��@�G�@�
=B  B�  ?��A
=@33AX  B���                                    Bx��c@  T          A  ?�{@���@}p�AᙚB���?�{A ��?���A33B���                                    Bx��q�  �          A
�\@�z�@���?Tz�@���B(�@�z�@�\)������\)B'�R                                    Bx�Ȁ�  
�          A
�R@θR@�?��@�(�B%33@θR@�ff�fff��\)B%�                                    Bx�ȏ2  
(          A(�@���@���?�
=A�RB1�@���@���&ff����B4(�                                    Bx�ȝ�  T          A��@��@ȣ�?�\)AD��B;�R@��@�G����
�B@p�                                    Bx�Ȭ~  �          A	��@�p�@�  @��AuG�B:ff@�p�@��>�?c�
BB                                      Bx�Ȼ$  T          A33@qG�@�  @��A�G�Bo�\@qG�@�G�?���AC
=B}                                    Bx����  T          A�@��R@��
?�ffAp�B;�@��R@�ff�Q����B=p�                                    Bx���p  �          A	@��@�=q����}p�B5p�@��@�p��
�H�l(�B-��                                    Bx���  �          A33@��H@���?�z�A�RBPQ�@��H@�(��\(����BQ�                                    Bx����  
�          A{@�33@�z�?�(�AY�Bg
=@�33@��;���L(�Bj�
                                    Bx��b  �          A=q@�ff@�  ?�A��B\�@�ff@ڏ\�p���θRB^                                      Bx��  T          A(�@XQ�A ��>��R@G�B�k�@XQ�@�p���H���B�z�                                    Bx��!�  T          A  @W�A z�>�ff@>�RB��@W�@��R��\�s�B��
                                    Bx��0T  �          A33@��@�{?��Az�Bup�@��@�{����
ffBuz�                                    Bx��>�  T          A  @��
@��@(�A�  BY��@��
@�\)���
��G�B`�                                    Bx��M�  T          A�@�ff@�(�?B�\@���Bl�R@�ff@�p����H�D��Bj                                      Bx��\F  �          A�@��\@�
=?�p�A/�Bdp�@��\@��k���33Bf\)                                    Bx��j�  �          A�
@�ff@�G�@33Atz�BW��@�ff@���#�
���B]p�                                    Bx��y�  T          AQ�@�ff@��@0  A�Q�Bfp�@�ff@���>8Q�?�
=Bm�                                    Bx�Ɉ8  
�          A(�@��@��?��HA.�\Bw�H@��@�p�������33By=q                                    Bx�ɖ�  �          A  @���@�  ?˅A#33BtG�@���@�=q��z���{Bu33                                    Bx�ɥ�  "          A��@�(�@�\)@z�AT  Bh�H@�(�@���!G��~�RBlQ�                                    Bx�ɴ*  T          A��@�33@�R?�33AF�HBo  @�33@���G����\Bq��                                    Bx����  T          A(�@��@�
=?�(�APQ�B[\)@��@�
=�\)�h��B_(�                                    Bx���v  T          A�
@�  @�=q?�Q�AM��B`ff@�  @�녿�R���\Bc�
                                    Bx���  
�          A@�@��@�At(�Bk�@�@����{��RBo�                                    Bx����  T          Aff@�  @��@�Av�HBi��@�  @����
��Bn�\                                    Bx���h  �          A��@�=q@�@��A|Q�Bf�@�=q@�=q�u��=qBk�                                    Bx��  T          Aff@���@�  @�A~=qBh(�@���@��;�  �˅Bm�                                    Bx���  �          A\)@u@�G�@	��A^ffB~��@uA �Ϳ333��Q�B���                                    Bx��)Z  �          A�\@|��@�p�@
�HAa�B{  @|��@�{�&ff��B~G�                                    Bx��8   
�          A=q@|��@�@ffAu��Bzff@|��@��R���>�RB~p�                                    Bx��F�  T          A�@W�@��H@
=Aw�B�k�@W�A�R�
=q�^�RB��                                    Bx��UL  T          Ap�@.{@�{@.{A�p�B�=q@.{A=q��  ��{B��                                    Bx��c�  T          Ap�@Mp�@�=q@Dz�A��B���@Mp�A�\>W
=?���B�                                      Bx��r�  
�          A�R@c�
@�(�@:=qA�Q�B�@c�
A�\<�>.{B��                                    Bx�ʁ>  �          A=q@x��@���@+�A�Q�Bz�R@x��@���#�
��G�B�{                                    Bx�ʏ�  T          A@�=q@�@{A���Bl�@�=q@�zᾀ  �У�Br�                                    Bx�ʞ�  �          A�
@�\)@Å@z�Ab�RB@Q�@�\)@�{�k���ffBFp�                                    Bx�ʭ0  T          A
=@���@�33@G�Av=qBI(�@���@�  ���c�
BO�H                                    Bx�ʻ�  
Z          A�R@=p�A\)?У�A)�B�k�@=p�A(���33�G�B���                                    Bx���|  "          A�@�A33?ǮA"{B��@�A33�Ǯ�!B��                                    Bx���"  T          A�?��A�?���@�{B��?��A
{�Q��X��B�p�                                    Bx����  �          A{@�Az�?���@���B���@�A	G�����ZffB�aH                                    Bx���n  T          A@"�\A
{?�=qA=qB�=q@"�\Az��{�<z�B��H                                    Bx��  T          A�@Z=qA\)?��
@ҏ\B��H@Z=qA z���R�MB���                                    Bx���  "          AG�@�=q@�33?��RA  Bhp�@�=q@�zῡG���Bi
=                                    Bx��"`  �          AQ�@���@���?�=qA&{B6z�@���@�녿@  ����B9Q�                                    Bx��1  T          AG�@�
=@�=q?�=qA$��B133@�
=@�\)�8Q����B433                                    Bx��?�  T          A{@�\)@�z�@ffAu�B-�@�\)@��H=u>\B6�                                    Bx��NR  �          Ap�@�p�@��
@p��A�B=(�@�p�@ڏ\?���A�
BN\)                                    Bx��\�  
Z          Az�@��H@�
=@eA��B6�R@��H@�z�?��\AG�BG��                                    Bx��k�  ^          A��@���@��
@N�RA�{BDff@���@�(�?J=q@�p�BQz�                                    Bx��zD  �          A33@�33@ڏ\@*�HA�G�BU(�@�33@��H        B\�H                                    Bx�ˈ�  
Z          A�@�G�@�ff@�RA��BT33@�G�@�z�\)�h��B[(�                                    Bx�˗�  T          A�@��R@�\?��
A�HBe{@��R@�(���G��
=Be��                                    Bx�˦6  T          A\)@��
@��@,��A�Q�B]  @��
@�׽u�\BdQ�                                    Bx�˴�  �          A	p�@���@���@!G�A���Be�H@���@��H�B�\���\Bl=q                                    Bx��Â  �          A	G�@��\@��H?�G�A=G�Bs�R@��\@�
=�����Bup�                                    Bx���(  "          A	��@_\)@�  @�\Aw\)B�
=@_\)@�녿!G���\)B���                                    Bx����  
�          A��@c33@�=q?�A@��B�33@c33@�ff��
=���RB��H                                    Bx���t  
�          A�@h��@�33@�AqB���@h��@���0�����B��                                    Bx���  
�          AG�@g�@��@\)A���B�u�@g�A �׿��Y��B��\                                    Bx���  J          A��@l��@��@G�Ao\)B�p�@l��@�ff�:�H��  B�#�                                    Bx��f  "          Az�@��@�z�@Tz�A�ffBDQ�@��@�ff?c�
@�G�BR                                    Bx��*  
�          A�@��H@�@/\)A���Bv��@��HAG���\)���B|z�                                    Bx��8�  �          A��@��@�@UA�z�B\�@��@�?�\@N�RBg��                                    Bx��GX  
�          A{@��@��@u�A�z�BK  @��@�(�?�z�@�G�BZz�                                    Bx��U�  
�          A{@�\)@�  @l(�A���BL@�\)@��?}p�@ǮB[(�                                    Bx��d�  
�          A��@��@ȣ�@���A�
=BA�@��@�=q?���Az�BS
=                                    Bx��sJ  �          A ��@��H@陚@O\)A�Q�BH�R@��H@�\)>��?�(�BR�                                    Bx�́�  T          A$  @��R@��@:=qA�{BR��@��RAG��u��=qBY�R                                    Bx�̐�  
d          A'
=@��A Q�@5A|Q�BS@��A  �Ǯ���BZ{                                    Bx�̟<  |          A'
=@�ff@�(�@:=qA��BO33@�ffA=q��\)�\BV33                                    Bx�̭�  T          A'�@�
=@�z�@=p�A�33BO
=@�
=A�H�u���BV=q                                    Bx�̼�  T          A'�@�(�@�ff@@  A�
=BQ�\@�(�A  �u���BX��                                    Bx���.  ,          A$(�@�\)@�(�@R�\A�BO�@�\)A��>��?Q�BY
=                                    Bx����  �          A�@�=q@�R@ ��Au�BQ(�@�=q@��
��
=�#33BWG�                                    Bx���z  ^          Az�@�(�@��?�{A>ffBI�@�(�@�  �^�R����BL�H                                    Bx���   
�          A  @�  @��
?�AD��BMG�@�  @�\�Y����z�BP�                                    Bx���  �          A�\@��H@�p�?�33AD��BH{@��H@�z�J=q���BK�R                                    Bx��l  T          Az�@�{@�33?��AG�BJ33@�{@ڏ\�G����\BM�                                    Bx��#  �          A  @�@���?ٙ�A2�HBK(�@�@�G��z�H��{BM�                                    Bx��1�  
�          A\)@��@��?��HAP(�B@ff@��@ҏ\�(���G�BE�                                    Bx��@^  T          A=q@�G�@�p�?�(�AMB?@�G�@��&ff��p�BD\)                                    Bx��O  �          A\)@��
@���?ٙ�A/33BIG�@��
@�������z�BKz�                                    Bx��]�  T          A�@���@�Q�?��
A4  BN��@���@�������ffBP�
                                    Bx��lP  T          A��@�(�@�33?��A+�BF�@�(�@�
=���
��
=BH33                                    Bx��z�  T          AG�@�G�@��@XQ�A��BCz�@�G�@��
?}p�@�33BT�                                    Bx�͉�  �          @�
=@�G�@���?�ffAS�
BC�@�G�@��׿z���ffBHG�                                    Bx�͘B  T          @��H@�@�p�?�ff@�Q�BSff@�@�33���(  BR{                                    Bx�ͦ�  �          @�=q@��H@�33?ǮA8��BT�@��H@�\)�p������BV\)                                    Bx�͵�  �          @�ff@�33@�{?�z�AffBEp�@�33@�p���  �p�BE{                                    Bx���4  T          @�
=@�  @�(�?�(�AMBAz�@�  @�33�(����BE�                                    Bx����  
�          @��@��H@�
=?Q�@���BV(�@��H@�G���
=�G�BS                                      Bx���  �          A�@�33@���>���@ffBeG�@�33@�33�Q����B^p�                                    Bx���&  �          Aff@�\)@޸R�#�
�uBj@�\)@����/\)���Bb{                                    Bx����  �          A  @�  @��
���Q�Bf@�  @�  �:=q���B](�                                    Bx��r  ^          A (�@�{@�{?�p�ARffBT�H@�{@�z�=p����BX�\                                    Bx��  
�          A�@��R@��þ�����Bb��@��R@�p��3�
���
BX�                                    Bx��*�  
�          @�@���@Ϯ������B\p�@���@���333���
BQG�                                    Bx��9d  �          @�=q@��R@��H�\)���
Be��@��R@�  �.�R��B[�
                                    Bx��H
  T          A�@��R@ָR>��@S�
B[@��R@�33�{�{33BU�R                                    Bx��V�  �          @�
=@�{@�(�>�@eBb�H@�{@�G���~{B](�                                    Bx��eV  �          @�ff@y��@��B�\����Brz�@y��@�G��;�����Bh�                                    Bx��s�  �          Ap�@R�\@�p���(��<��B��\@R�\@ۅ�\(����HB���                                    Bx�΂�  S          A	p�@tz�@����\)���B}�H@tz�@���S�
���Bt33                                    Bx�ΑH  T          Az�@�=q@�Q���[�By=q@�=q@����e����\Bm�
                                    Bx�Ο�  �          A�@\)@��Ϳ
=q�aG�B|@\)@����j=q���HBq�                                    Bx�ή�  �          A
=@s�
@�Q���Ϳ(��B|Q�@s�
@�33�Fff����Bsz�                                    Bx�ν:  �          @�ff@�@��?E�@��Bg�\@�@�����H�fffBc��                                    Bx����  �          A
�R@qG�@��ÿ��`  B�Q�@qG�@���g
=���Bu��                                    Bx��چ  �          A��@{�@��H>�
=@/\)B}�\@{�@�33�1G���G�Bwp�                                    Bx���,  �          A
�H@o\)@��?p��@ȣ�B�p�@o\)@�R����r{B}�\                                    Bx����  �          Az�@n{@�G�?���A�HB{@n{@�{���G\)B}                                    Bx��x  �          A	�@mp�@�G�?У�A0(�BG�@mp�@�녿Ǯ�'�Bz�                                    Bx��  �          A33@�=q@�p�?�{AK33Bq�R@�=q@陚�����33Bs�\                                    Bx��#�  T          A  @�p�@�z�?�Q�A-�Bo��@�p�@�p���ff�
=Bp33                                    Bx��2j  �          A33@�
=@��H?��RA��Bm��@�
=@�G���(��1�BmG�                                    Bx��A  �          A��@�@�Q�?s33@���Bb��@�@��(��`z�B_
=                                    Bx��O�  �          A\)@��\@�=q?\(�@�Q�B\��@��\@�������c�BX��                                    Bx��^\  "          A  @��H@�\)?xQ�@�
=BM(�@��H@أ׿��H�IG�BI�R                                    Bx��m  T          A�
@���@�z�?Tz�@���BSp�@���@�33�
=q�^ffBN��                                    Bx��{�  T          A@�{@�\)?xQ�@ÅBT(�@�{@�  ���S33BPz�                                    Bx�ϊN  �          A�@��H@�=q>�33@(�BR��@��H@�=q�+���\)BJ�R                                    Bx�Ϙ�  �          Az�@��R@�
=?0��@�Q�BWQ�@��R@��H�(��t  BQ��                                    Bx�ϧ�  �          A(�@�=q@�  ?u@�
=BZ��@�=q@�\)�p��]G�BV��                                    Bx�϶@  �          A�\@��@��H?333@��BA@��@У����\��B<G�                                    Bx����  T          A�@�@���?�G�@���BP{@�@��Ϳ�{�8  BN{                                    Bx��ӌ  �          A  @�@���?�p�A�BN=q@�@��
��{�   BM��                                    Bx���2  �          A\)@�Q�@�\)?�p�AQ�BR��@�Q�@�{����$z�BRG�                                    Bx����  T          A(�@���@�Q�?ǮA33BR��@���@�  �˅�=qBR�R                                    Bx���~  T          A��@�\)@�33?�p�A�\BU(�@�\)@陚���H�)��BTff                                    Bx��$  T          A��@��@�ff?�\)A
=BX{@��@�33��{�8  BVz�                                    Bx���  T          Az�@���@�{?��R@�{BX(�@���@��ÿ�(��DQ�BU��                                    Bx��+p  �          A��@��@��?G�@��HBZ��@��@�{�(��s\)BUQ�                                    Bx��:  T          A(�@��
@陚?aG�@�ffBQ��@��
@�  �  �ap�BL��                                    Bx��H�  �          A33@�{@�\)?\(�@��RBS
=@�{@���ff�f�HBN{                                    Bx��Wb  T          A�@Å@�p�?k�@�33BFG�@Å@�z��
�H�S�BA�H                                    Bx��f  �          A{@Å@�>��?ǮBE(�@Å@���0  ��z�B<                                      Bx��t�  T          A�@���@��>#�
?xQ�BG�@���@���7
=����B=                                    Bx�ЃT  �          A=q@��@�\)>���@��BS�\@��@�ff�2�\��{BKp�                                    Bx�Б�  �          A  @���@��?��RAAG�Bd�
@���@�p����R�z�BfG�                                    Bx�Р�  �          A�@�\)@�z�?���A8  Bc��@�\)@��R������Bd�\                                    Bx�ЯF  T          A{@��R@�=q?�Q�AD(�Bh\)@��R@�p���Q��Bi�
                                    Bx�н�  �          A�H@�G�@���?���A733Bn@�G�@�녿���$��BoG�                                    Bx��̒  �          A�R@���@��R@G�AeG�Bq(�@���@�{��(���(�Bt{                                    Bx���8  �          A=q@��
@�Q�@!�A��RBx
=@��
A����  ����B|{                                    Bx����  
�          A�@{�@�z�@Dz�A��B{�@{�A����?\)B��H                                    Bx����  �          A��@�@��
@:�HA��Bk�@�@���   �J=qBr��                                    Bx��*  T          A(�@��@�  @(Q�A��Br{@��@���Q���
=Bw(�                                    Bx���  �          A
=@��@�>#�
?��Bgp�@��@����E�����B]�                                    Bx��$v  �          A33@��
@���#�
�}p�Bk�R@��
@ۅ�Z�H��B`=q                                    Bx��3  �          A=q@�ff@��>u?�ffBoQ�@�ff@���Dz����Bf�                                    Bx��A�  T          A  @��H@�
=>�p�@��Bxff@��H@�(��>{���Bp                                    Bx��Ph  �          A
�\@L��A Q쾞�R��B�� @L��@��n{��(�B��\                                    Bx��_  "          A\)@��
@�{?��A-B:33@��
@�G���Q����B<
=                                    Bx��m�  T          AQ�@��@�G�?n{@ȣ�Ba�H@��@�  ����o�
B]z�                                    Bx��|Z  "          AQ�@�@�{?�Q�A   Biff@�@߮���\z�Bf��                                    Bx�ы   �          A(�@�z�@�G�?��HAffBbff@�z�@��
�����T  B_�
                                    Bx�љ�  �          AG�@{�@�p�?���A-p�Bu33@{�@�z��Q��;�Bt��                                    Bx�ѨL  �          A��@Vff@��
?�p�A@��B��@Vff@�z����6ffB�8R                                    Bx�Ѷ�  T          A��@l(�@�(�?u@�(�B}�R@l(�@�������By�                                    Bx��Ř  T          A��@~�R@�R?z�H@���BtG�@~�R@�p��  �{�
Bp=q                                    Bx���>  �          A  @�33@߮?�
=A ��BnG�@�33@����  �DQ�Bm=q                                    Bx����  �          Az�@XQ�@�
=?��RA��B���@XQ�@�\)�  �u�B�W
                                    Bx���  
�          A
{@r�\@�ff?fff@���B33@r�\@�=q�#33��G�Bz�\                                    Bx�� 0  
�          A
�\@\(�@�z�?z�H@�\)B��)@\(�@���$z���  B��
                                    Bx���  �          A
=q@Mp�@�{?�\)@�{B��@Mp�@�(��{���HB�L�                                    Bx��|  "          A
=q@]p�@�=q?�33@�B�B�@]p�@�G��������B���                                    Bx��,"  T          A
{@j=q@���?W
=@�33B��{@j=q@�33�)����G�B~(�                                    Bx��:�  
�          A	�@y��@�33?:�H@�z�B{\)@y��@�z��+����Bu�\                                    Bx��In  
�          A�R@n�R@�?W
=@��B~(�@n�R@��H�"�\����By(�                                    Bx��X  "          A33@�{@љ�@�\Ag\)B_p�@�{@��ÿ�ff���Bc{                                    Bx��f�  �          Ap�@�\)@ə�@Q�A��HBZQ�@�\)@ָR�&ff��=qB`��                                    Bx��u`  �          A ��@�G�@ȣ�@(��A��B^Q�@�G�@�G������8Q�Bf�R                                    Bx�҄  �          A Q�@��@\@.{A�{BV=q@��@���������B_��                                    Bx�Ғ�  �          @��R@��@\@0��A��\BZ��@��@��aG��˅Bd�\                                    Bx�ҡR  T          A Q�@�{@�ff@�\A��RBc�
@�{@�G��O\)����BiG�                                    Bx�ү�  �          Az�@�(�@���@�
A�Bj�@�(�@�33�p���љ�BoQ�                                    Bx�Ҿ�  �          AQ�@��@�33@Q�A��
BbQ�@��@�
=�O\)���Bg��                                    Bx���D  �          A�@�{@�\)@��A��B^33@�{@��
�=p���p�Bd\)                                    Bx����  �          A\)@���@��
?�p�ABffB^�@���@�ff��z��{B_�R                                    Bx���  �          AQ�@�(�@���?��A
=BU
=@�(�@�
=���:�RBS�                                    Bx���6  �          @��\@���@��?p��@�{BFz�@���@��R��ff�T��BB�\                                    Bx���  �          @�  @�  @��@�A�G�BM{@�  @��R��R��\)BS�H                                    Bx���  �          @���@��@�=q?�(�AT(�BFff@��@�  �u���BJ                                      Bx��%(  �          @�  @��R@�\)?���AQ�B4p�@��R@������+�
B2                                    Bx��3�  �          @�@j=q@�G�?�\)A��BX{@j=q@��H�����  B^\)                                    Bx��Bt  �          @�33@8��@��R@z�A�z�Bx\)@8��@�p��Ǯ�W�B�
                                    Bx��Q  �          @�z�@0  @�ff@�HA��RBx��@0  @�\)�B�\��
=B���                                    Bx��_�  T          @�?�Q�@qG�?���A�ffB�\?�Q�@�{��\)�uB��                                    Bx��nf  �          @���?�33@h��?�  A�z�Bz  ?�33@{����R��G�B�p�                                    Bx��}  �          @�33@(�@aG�?�p�A�=qBhz�@(�@l�Ϳ   �ҏ\Bmff                                    Bx�Ӌ�  �          @r�\@�@:�H?O\)AE�BP{@�@>{�����BQ�                                    Bx�ӚX  �          @~�R?�33@Z�H?z�HAf�HB��\?�33@_\)�.{�   B�u�                                    Bx�Ө�  �          @�33@p�@_\)?�\)A��Bf@p�@n�R��Q�����Bm��                                    Bx�ӷ�  �          @��
@�@w
=?�(�A��HBx��@�@����ff��=qB~�                                    Bx���J  �          @��@ff@q�?�\A�  Bs�@ff@�p��\)��\B|�H                                    Bx����  
�          @�(�?�z�@c�
?޸RA�Q�Bwp�?�z�@}p����
�z�HB���                                    Bx���  �          @���@Q�@s33?�\)Af�\Br�R@Q�@x�ÿ=p��
=Bu
=                                    Bx���<  �          @�G�@&ff@u?���A\  B_�@&ff@|(��@  ��
Bb=q                                    Bx�� �  
�          @���@@��@��\?�
=A|��BVQ�@@��@��ÿ
=��{B[��                                    Bx���  "          @�Q�@S�
@|(�?��A>�\BG�@S�
@�  �W
=��HBI\)                                    Bx��.  �          @�z�?���@\(�?��An�RBxz�?���@a녿(���{B{                                      Bx��,�  T          @W�?�{@�H?�G�A���B��?�{@6ff>k�@��\B��=                                    Bx��;z  T          @W
=@   @
=?�p�A��HBG�@   @*=q    =#�
BTp�                                    Bx��J   
�          @}p�@�@9��?�A���BY�@�@U>��@
�HBh�\                                    Bx��X�  �          @��@
=@E?�33A�{B^p�@
=@`  =#�
?
=qBkz�                                    Bx��gl  �          @Mp�?�{@
�H?�{A���BV�?�{@*�H>�(�@�=qBk��                                    Bx��v  
�          @Q�?�
=@
=?�B��BN
=?�
=@.{?&ffA8��BiQ�                                    Bx�Ԅ�  �          @e?�\?�@��B)�HB=Q�?�\@3�
?�p�A���Bg�                                    Bx�ԓ^  �          @���@�R?��@Y��BR��A�@�R@/\)@=qB	(�BKQ�                                    Bx�Ԣ  �          @|��?�=q�B�\@1G�B���C���?�=q?(�@3�
B��A�=q                                    Bx�԰�  �          @�G�@z�@�Q콏\)�^�RBz�
@z�@aG����ɮBn=q                                    Bx�ԿP  �          @��@7
=@�{>��?�\)Bd�@7
=@�  ��
=��p�BY��                                    Bx����  �          @��\@.{@�\)=�Q�?���Bd@.{@q녿�����BX��                                    Bx��ܜ  �          @���@4z�@tz�>�Q�@��BVG�@4z�@c33���H����BN(�                                    Bx���B  �          @�G�@,(�@e?�R@��\BT�@,(�@^{�����f�HBP�R                                    Bx����  
�          @��
@!G�@Y��?k�ADz�BU�@!G�@\(��E��$(�BV�
                                    Bx���  
�          @��H@1�@J�H?xQ�AO�BCG�@1�@P�׿�R���BFQ�                                    Bx��4  �          @�(�@=q@C33?�{A\)BN�R@=q@Mp���G����BT�\                                    Bx��%�  �          @���@7�@G�?ǮA��
B=p�@7�@_\)���
���
BJ�                                    Bx��4�  �          @�Q�@.�R@K�?�z�A�z�BE�\@.�R@e���\BR��                                    Bx��C&  T          @�  @2�\@  @.{B�RB�\@2�\@QG�?���A�
=BE�H                                    Bx��Q�  "          @��@S�
@N�R?�Q�A��
B1=q@S�
@p��>B�\@(�BB�                                    Bx��`r  T          @��\@g
=@(�@33A�p�Bp�@g
=@G�?(��@��B#ff                                    Bx��o  T          @�Q�@o\)@�
@�A�\)A�33@o\)@HQ�?xQ�A3�B�                                    Bx��}�  �          @�(�@g
=@.{?���A��BG�@g
=@P��>��
@tz�B(�\                                    Bx�Ռd  
�          @��\@^�R@E�?�p�Amp�B%��@^�R@R�\��p���z�B-�R                                    Bx�՛
  �          @�{@\(�@`��?�Q�A��\B633@\(�@qG��Ǯ���
B>z�                                    Bx�թ�  �          @�  @<��@E@A���B9��@<��@n{>\@�=qBNff                                    Bx�ոV  �          @�  @(�@Y���u�L��BY33@(�@=p���z���
=BJ{                                    Bx����  �          @�ff@�@`  ���
��(�Bc�@�@;���Q���
=BP�                                    Bx��բ  �          @�\)@G�@j�H=u?G�Bt�@G�@P  ��Q���z�Bh��                                    Bx���H  "          @�z�@!G�@O\)>��R@���BPQ�@!G�@?\)���\���BG�\                                    Bx����  �          @��\@'�@A�?   @��
BD�@'�@:=q��  �eB?��                                    Bx���  T          @s�
@33@E�?(��A!�Ba��@33@AG��c�
�[33B_�                                    Bx��:  �          @Q�?�p�@,��?.{A@  Be��?�p�@,�Ϳ0���B{Be�                                    Bx���  
�          @Y��?�\@1G�?!G�A,(�Be��?�\@/\)�E��UBdff                                    Bx��-�  �          @*=q?��
?�?�33B �HBZ��?��
@��>�Q�@�
=Bq�                                    Bx��<,  �          @e?�@(��?��A���BW��?�@;����
=Bc\)                                    Bx��J�  �          @�=q@�@#�
?�A���B<z�@�@G
=>�Q�@���BR�                                    Bx��Yx  �          @��?�\)@z�H?�Av�\B�\?�\)@�  �Y���2�RB��H                                    Bx��h  �          @���?�\@�ff@
=qA�(�B��R?�\@�{�u�(Q�B���                                    Bx��v�  T          @��H?��
@�p�@)��A��
B���?��
@���>W
=@{B�                                      Bx�օj  T          @�ff?�
=@��@"�\Aޣ�B��R?�
=@��H=�G�?��B�u�                                    Bx�֔  �          @�\)@�@��@'
=A��B\)@�@���>W
=@��B�aH                                    Bx�֢�  �          @�\)?�\@�{@B�\B�B�G�?�\@�z�?&ff@ٙ�B�
=                                    Bx�ֱ\  _          @�{?�\@���@7�A��RB���?�\@�(�>�@���B��f                                    Bx���  I          @��
@�\@���@:=qA���B��\@�\@�\)>��R@A�B��                                    Bx��Ψ  
Z          @���?�{@���@=p�A�B�p�?�{@�
=>k�@��B�                                      Bx���N  �          @��H?�  @�z�@5�A��
B��
?�  @���=�G�?�\)B�8R                                    Bx����  
�          @���?�ff@��@@  A��HB�p�?�ff@�=q>W
=?��RB��                                    Bx����  
�          @�
=>�
=@�z�@N�RB�
B�33>�
=@���?333@�\B�8R                                    Bx��	@  "          @��\?G�@�
=@Mp�B{B�G�?G�@��R?#�
@��HB���                                    Bx���  T          @���?�G�@�G�@UB\)B�W
?�G�@���?uA$��B��=                                    Bx��&�  
Z          @�  ?�\)@p  @1�B�B�G�?�\)@�(�?��@߮B�ff                                    Bx��52  �          @�\)@�H@{�@z�A�G�Bip�@�H@��=�G�?�  Bx�                                    Bx��C�  �          @�p�@`  @o\)@��A�=qB;@`  @��
>��?��
BM=q                                    Bx��R~  �          @�z�@�z�@��R@\)A�G�B5�@�z�@�z�>�?���BG
=                                    Bx��a$  T          @�
=@���@�z�@"�\A��RB0\)@���@�33>B�\?�(�BBz�                                    Bx��o�  "          @�{@�(�@��@=qA��
B8�@�(�@�{    ��BHff                                    Bx��~p  �          @�33@�33@��?�33Aip�B!(�@�33@�z����{B(                                    Bx�׍  T          @�33@G�@���@&ffA�\)BQ��@G�@���>�\)@4z�Bc��                                    Bx�כ�  
�          @�
=?���@s33@j�HB"�
B{\)?���@��?��A\��B�B�                                    Bx�תb  �          @�p�?xQ�@R�\@�{BQ=qB��?xQ�@���@��A�B�33                                    Bx�׹  
�          @�
=��G�@8Q�@���Bl�HB�8R��G�@��H@K�A��BϽq                                    Bx��Ǯ  �          @��
�Y��?��H@�z�B�W
B��ÿY��@qG�@Mp�B��B�p�                                    Bx���T  �          @�  �#�
@\)@�B|p�B�#׾#�
@�(�@0��B��B�k�                                    Bx����  �          @�z�?��@0��@�{B~\)B�\)?��@�ff@eB	�HB�Ǯ                                    Bx���  T          @�z�?�Q�@Vff@�33Blp�B�\)?�Q�@�p�@j=qA��
B��{                                    Bx��F  .          @�  @=q@mp�@�G�BW(�Bd33@=q@ƸR@[�A��B���                                    Bx���  
�          @��@O\)@w�@�{B<��BHz�@O\)@���@4z�A���Bwz�                                    Bx���  �          @�@\(�@�@��B.z�BJG�@\(�@�@�HA�33Bs�                                    Bx��.8  T          @�{@��R@�(�@�B��B+��@��R@��
?�(�A0z�BM                                      Bx��<�  T          A�R@�z�@��R@�z�A��B�@�z�@�ff?\A+�B5\)                                    Bx��K�  
�          A ��@�
=@l��@y��A�Q�A�ff@�
=@���?ǮA2�RB#
=                                    Bx��Z*  �          A@�Q�@��@p��A���B@�Q�@��
?z�H@�{B;G�                                    Bx��h�  
�          @�ff@�G�@�=q@W�A��B{@�G�@�z�?(��@���B6
=                                    Bx��wv  "          A�@�(�@�(�@q�A��B��@�(�@���?z�H@�33B9Q�                                    Bx�؆  �          Az�@�@��H@G�A�{B�@�@���>�
=@6ffB"��                                    Bx�ؔ�  �          A
{@��@�ff@1G�A��A�
=@��@���>��
@ffBG�                                    Bx�أh  �          A	�@�{@j�H@>�RA��A�ff@�{@�p�?@  @���B�                                    Bx�ز  T          A
�R@�Q�@Mp�@_\)A��
A���@�Q�@���?�
=A�A�G�                                    Bx����  �          A
=@�\)@Fff@k�Aȏ\A��@�\)@���?��A.�\B 33                                    Bx���Z  �          A33@ۅ@O\)@�  A��
A�{@ۅ@�p�@�
A[�B��                                    Bx���   �          A33@�{@Vff@�{A�33A�z�@�{@��@
�HAg\)B�                                    Bx���  �          A�@޸R@K�@�A�G�Aģ�@޸R@�33@�AW�Bff                                    Bx���L  �          A
�R@�(�@2�\@�ffA�p�A���@�(�@���@p�A��
B{                                    Bx��	�  T          A
�R@�\)@XQ�@��B(�A�=q@�\)@�Q�@ffA{33B33                                    Bx���  �          A�@�
=@J=q@�z�B��A�Q�@�
=@�@)��A��B�H                                    Bx��'>  �          A��@���@Q�@�ffB�A�p�@���@�p�@S�
A�G�B=q                                    Bx��5�  �          A�@�ff@3�
@��B�A�ff@�ff@��
@2�\A���B(�                                    Bx��D�  
�          Ap�@��@c33@��HA�A�=q@��@�  ?�(�ANffBp�                                    Bx��S0  T          A{@ָR@�z�@�G�A���A�\)@ָR@��?�z�AffB�                                    Bx��a�  T          Az�@�{@��R@�=qA���B��@�{@��?��\A{B%�R                                    Bx��p|  �          A
�R@�  @���@��A�  A�z�@�  @���?\A!p�B"
=                                    Bx��"  �          A
=@���@��\@x��A�B��@���@�\)?��A�RB$z�                                    Bx�ٍ�  T          A�@�  @�p�@z=qA�  B�@�  @�=q?��\A
�\B&��                                    Bx�ٜn  
�          A
=@\@�ff@s33A�=qB�
@\@�Q�?��@�=qB-�
                                    Bx�٫  �          A��@�G�@��@�
=A��B�@�G�@Å?���A�\B:
=                                    Bx�ٹ�  T          Az�@�\)@y��@��\A�{B {@�\)@���?�\A?\)B&                                    Bx���`  �          A	��@Ǯ@k�@��\B33A�(�@Ǯ@��@Aa�B%\)                                    Bx���  
�          Az�@��\@�p�@���A�ffB��@��\@�G�?ǮA(z�B8(�                                    Bx���  �          A	p�@�(�@��@�A�\)Bff@�(�@��\?�A��B.Q�                                    Bx���R  "          A��@�p�@���@x��A�(�B(�@�p�@��R?�ffA33B&{                                    Bx���  T          Aff@�33@xQ�@x��A���A�G�@�33@��?��A  B�R                                    Bx���  T          A�H@�z�@���@~{AᙚB{@�z�@��H?���Az�B)33                                    Bx�� D  �          A�\@���@w
=@���A�=qBp�@���@��?�=qAH��B*�\                                    Bx��.�  �          Aff@�
=@b�\@��A��A���@�
=@�G�@   A\Q�B!ff                                    Bx��=�  �          AQ�@\@�  @�ffA�{B�
@\@���?���A�B.�                                    Bx��L6  �          A
=@�  @��R@`  A�ffB#�H@�  @ȣ�>�Q�@=qB9
=                                    Bx��Z�  �          A
=q@���@�Q�@fffA�p�B�@���@���?��@h��B633                                    Bx��i�  `          A�
@�ff@�Q�@qG�A�G�B%\)@�ff@�\)?0��@���B>G�                                    Bx��x(  �          A�@�(�@�\)@[�A�
=B6
=@�(�@�<�>B�\BGz�                                    Bx�چ�  �          A
{@�Q�@�G�@/\)A�=qB*��@�Q�@�p�����I��B6��                                    Bx�ڕt  �          A@ȣ�@�=q@Tz�A��B!33@ȣ�@�Q�>�?^�RB3�                                    Bx�ڤ  �          A  @�=q@�(�@b�\A�G�B�R@�=q@�  ?��@fffB.(�                                    Bx�ڲ�  �          Ap�@���@�33@l��A��
A���@���@��?�=q@���Bff                                    Bx���f  �          Aff@޸R@|��@y��A�  A�z�@޸R@�z�?�=qA	��B                                      Bx���  �          A
=@�=q@��@s33A�
=B(�@�=q@�Q�?W
=@��
B)�H                                    Bx��޲  T          A�@љ�@�p�@�  A�\)B@љ�@�  ?�G�AG�B%�                                    Bx���X  �          A{@�Q�@fff@���A��HAم@�Q�@��?�\)A(Q�Bp�                                    Bx����  �          A��@�33@(�@�Q�A�ffA��R@�33@�=q@!�A�=qA癚                                    Bx��
�  �          A
{@أ�@\��@�=qA�p�A�@أ�@���?�(�A8��B��                                    Bx��J  �          A	��@�(�@?\)@k�A˙�A�=q@�(�@��R?�\)A.ffB {                                    Bx��'�  T          A	�@�z�@8Q�@O\)A��A�  @�z�@�(�?�ffA  A�                                    Bx��6�  �          A
=q@��@%�@C33A�p�A���@��@q�?��A	G�A�{                                    Bx��E<  �          A	��@��@<(�@W�A�p�A��@��@�Q�?���A  A�                                      Bx��S�  T          A��@��@X��@�=qA��A�z�@��@�(�?��HAS�B{                                    Bx��b�  �          A�R@�
=@^{@�ffB �A�G�@�
=@���@�A_�B!�                                    Bx��q.  `          A�@Ǯ@��R@�Q�A�=qB�@Ǯ@�p�?�G�AQ�B)
=                                    Bx���  �          A�@��@5@�=qA�G�A��@��@�G�@G�Ag
=B=q                                    Bx�ێz  �          A�@��R@#33@��BffA��@��R@�Q�@=p�A��Bff                                    Bx�۝   �          A��@��@^�R@��RA�Q�A���@��@���?�AH(�B�H                                    Bx�۫�  �          A(�@�z�@vff@��\A�ffA��R@�z�@��?�
=AK�
B&=q                                    Bx�ۺl  �          A�@���@~�R@�\)Bp�B�@���@�33?��RAS
=B.\)                                    Bx���  �          A�@���@�{@�{B�B
=@���@�z�@�
AW33B6                                      Bx��׸  �          A
=@�ff@��
@��A�
=B@�ff@ȣ�?���A��B5\)                                    Bx���^  �          A��@�(�@��
@�
=B{Bz�@�(�@�z�?�33A+�B=�\                                    Bx���  �          A��@��H@��R@��HA��Bz�@��H@��?�p�A�B>�\                                    Bx���  �          Ap�@��@�33@��A�RB&z�@��@��
?�=q@�G�BDQ�                                    Bx��P  T          A��@�ff@�p�@�(�A�p�B�R@�ff@��R?�\)A33B*�\                                    Bx�� �  �          A��@љ�@~{@mp�A��HA�  @љ�@�=q?�{@�Bp�                                    Bx��/�  T          A	�@��R@�=q@�z�A��\B��@��R@�
=?�G�A"ffB4�                                    Bx��>B  �          A	�@�  @�
=@z=qA��B\)@�  @�33?��\@���B,�                                    Bx��L�  �          A
=@�p�@��\@j=qA�B@�p�@�z�?�G�@�33B {                                    Bx��[�  �          AQ�@ə�@\)@c�
A�{Bp�@ə�@�Q�?u@�Bp�                                    Bx��j4  �          A��@�33@���@B�\A�BG�@�33@��>�p�@(Q�B"�                                    Bx��x�  �          A ��@��@r�\@E�A��A���@��@��H?(��@�p�B\)                                    Bx�܇�  
�          @���@�  @z�H@8��A�B@�  @��>�
=@HQ�B��                                    Bx�ܖ&  �          @���@�  @u@>{A�(�B�
@�  @�=q?
=q@��\B��                                    Bx�ܤ�  
�          @���@��H@s�
@0  A�  B��@��H@�>�Q�@8��B%                                      Bx�ܳr  �          @޸R@��H@G�@<(�A�33A��
@��H@^{?�=qA0��A�                                    Bx���  "          @�\@�{@w
=@5�A�33Bz�@�{@���>���@P  B*G�                                    Bx��о  .          @�ff@��@qG�@=p�A�p�B�R@��@���?\)@�=qB!p�                                    Bx���d            @��@��@hQ�@6ffA���B@��@��\?�@���B
=                                    Bx���
  T          @�\)@��H@y��@\��AܸRB�R@��H@�(�?fff@�ffB/�\                                    Bx����  �          @��@��@u@i��A癚Bff@��@�?���A=qB0{                                    Bx��V  �          @�=q@�Q�@E�@s33A�A��@�Q�@��
?�\)AF�\B��                                    Bx���  T          @��
@�  @`��@X��A��B�@�  @�Q�?�ffA�
B#��                                    Bx��(�  �          @���@�33@��H@.�RA���B-��@�33@��H���
�+�B>�H                                    Bx��7H  �          @�p�@�=q?�G�@9��A���Ap(�@�=q@AG�?�ffAAp�AŮ                                    Bx��E�  
�          @�33@θR?�z�@Dz�A�Af�R@θR@AG�?޸RA[�AȸR                                    Bx��T�  T          @�p�@�ff?޸R@H��A�(�A{�@�ff@HQ�?�G�Ac�A֣�                                    Bx��c:  �          @�Q�@�
=@,(�@<(�AǅA�p�@�
=@u�?��A  BG�                                    Bx��q�  �          @��@��@��?�z�Ai��B���@��@�
=��p��s�B��=                                    Bx�݀�  T          @�G�?�=q@�\)?�(�AQ��B�aH?�=q@\�G���33B��H                                    Bx�ݏ,  �          @�  @:�H@��
@   A��Bq{@:�H@��Ϳ}p��ffBvff                                    Bx�ݝ�  .          @Ӆ@\)@�{?��A5��B��\@\)@�\)������B�{                                    Bx�ݬx  z          @�G�@�\@�(�?�ffA:�HB�k�@�\@�{��p����B��                                    Bx�ݻ  T          @�G�@.{@���?�z�A@��B�ff@.{@��
��(�����B�=q                                    Bx����  �          @ҏ\@��@Å?��A z�B�Q�@��@�G��\)����B�ff                                    Bx���j  "          @˅@ff@�?G�@�33B�  @ff@�ff�p����\B��H                                    Bx���  T          @�G�@��@���?�@�Q�B�p�@��@�\)�$z���(�B�G�                                    Bx����  �          @�p�@J�H@���?�=qAT  Bw(�@J�H@�33��\�n�RBvG�                                    Bx��\  �          @ڏ\@>{@���?z�HA��BQ�@>{@���
=���
By(�                                    Bx��  �          @�@�@ڏ\?��
A33B��
@�@�33�.�R��Q�B�u�                                    Bx��!�  
Z          @��@   @�33?#�
@��HB�(�@   @˅�Mp�����B�\                                    Bx��0N  �          @�z�@G�@�Q�>���@G
=B�(�@G�@���W
=����B���                                    Bx��>�  
�          @�{?�p�@�R>��H@r�\B�\?�p�@�(��X����G�B��R                                    Bx��M�  T          @��?��
@�z�?Q�@�(�B�?��
@Ϯ�Dz���Q�B�\)                                    Bx��\@  �          @�{?�33@�G�>�  ?�z�B�=q?�33@�=q�i����
=B�                                      Bx��j�  
�          @�@�\@ᙚ>�=q@ffB���@�\@��
�`  �㙚B�{                                    Bx��y�  �          @��?�@�>��R@z�B�� ?�@���h����(�B��H                                    Bx�ވ2  �          @�?�=q@�p�>���@��B�=q?�=q@����Z=q��p�B��{                                    Bx�ޖ�  �          @�33?G�@أ׾k�����B��?G�@��
�q���RB�G�                                    Bx�ޥ~  �          @�\)>�z�@���
=���\B��>�z�@��\������B��H                                    Bx�޴$  
�          @�\)���@�����33B�  ���@����]p��Q�B�                                    Bx����  �          @���.{@��
�Q��	p�B�p��.{@��
�b�\�!�Bƽq                                    Bx���p  �          @��
��33@��Ϳ�Q��2�\B����33@��
��(��)�Bя\                                    Bx���  �          @�녿&ff@��H����^�HB�\�&ff@Fff�N{�5��B˳3                                    Bx���  T          @�z��(�@�z΅{���\B��=��(�@Mp��g��@�RB�.                                    Bx���b  �          @��H��\)@s�
�:�H�%��B��f��\)@5�(���#{B�#�                                    Bx��  �          @�=q�0��@vff�k��S
=B�k��0��@1G��4z��2\)B�\                                    Bx���  
�          @����
@�z΅{��ffB�B���
@1��Vff�+\)C�f                                    Bx��)T  �          @�(��}p�@��H���\�q�B�G��}p�@N{�aG��8��B�#�                                    Bx��7�  T          @���!G�@\)�^�R�A�B���!G�@:�H�7
=�.�
B�.                                    Bx��F�  �          @��H���@����=q��B�B����@Fff�tz��9z�B�Ǯ                                    Bx��UF  �          @���� ��@�=q�У�����B�G�� ��@Mp��|(��8p�B�                                    Bx��c�  �          @����\@��H�������B�Q��\@l����(��L�BÏ\                                    Bx��r�  T          @�\)���\@�33�ٙ�����Bȳ3���\@u����@{B��
                                    Bx�߁8  �          @�(����
@��
����ffB͸R���
@�ff�^�R��B�#�                                    Bx�ߏ�  "          @�p���Q�@�(���
��p�B؊=��Q�@p  ���\�I{B�q                                    Bx�ߞ�  "          @��
�(Q�@�33�[����
B�\)�(Q�@:�H��(��a
=C�                                    Bx�߭*  T          @�z��#33@����R�\��\)B�p��#33@J=q���
�]\)C ��                                    Bx�߻�  �          @�G���R@����P  ��
=B�\��R@J�H���H�a��B�#�                                    Bx���v  T          @�R�+�@�(��e���
=B����+�@7
=�����d  C5�                                    Bx���  �          @�ff�1G�@�p����
�Q�B�G��1G�@���  �r=qC�                                     Bx����  .          @�\�HQ�@�������\B�#��HQ�?�ff��33�qQ�C)                                    Bx���h  �          @�p��N�R@���Q����B�#��N�R?��У��vz�C5�                                    Bx��  �          @��^{@u�����;�C\�^{>�z��׮�z��C/+�                                    Bx���  �          @�p��h��@dz����R�A�C���h�ü��׮�v�C4z�                                    Bx��"Z  
�          @�33�g�@�
=��p��+��C�H�g�?O\)���
�r��C'\)                                    Bx��1   �          @�z��y��@�������Cu��y��?s33��ff�dz�C&Q�                                    Bx��?�  T          @�\)�O\)@l(���  �#ffCB��O\)?\(�����kC%33                                    Bx��NL  
(          @�p��a�@h����G��!  C)�a�?J=q���\�d{C'n                                    Bx��\�  �          @�Q��j=q@n{�p  ���C�\�j=q?������Up�C"��                                    Bx��k�  �          @�
=�C�
@>{��
=�?�\C�)�C�
�#�
���\�t��C4(�                                    Bx��z>  �          @��;�@�p�����Ə\B�(��;�@���{�EffCk�                                    Bx����  
�          @����U�@�{�1����B��U�@�R��33�E�CJ=                                    Bx����  
�          @ʏ\�>{@�z��%���B�u��>{@0  ����G�C	+�                                    Bx��0  �          @�p���z�@�Q��Z�H�\)B��
��z�?Ǯ�������C�f                                    Bx���  "          @����,(�@S�
��R��C(��,(�@p��33� �C	��                                    Bx���|  �          @��H� ��@h�ÿ�\)��33B�G�� ��@��aG��<�Cp�                                    Bx���"  �          @��
�{@W��x���/�
B��
�{?E����Hk�C �q                                    Bx����  �          @�
=��\@c�
��Q�����B�k���\?�(��c33�F=qC:�                                    Bx���n  �          @��
��(�@\���<����\B�  ��(�?�����H�}33C�                                    Bx���  �          @��\��
=@c33�tz��-z�B�.��
=?u���
W
C�3                                    Bx���  �          @���aG�@u�0  ��
B�녿aG�?�\��(�ffB�
=                                    Bx��`  �          @��׿.{@��\�3�
�	
=B���.{?�Q�����=B�u�                                    Bx��*  �          @�p����\@|���U�=qB׮���\?�ff��CY�                                    Bx��8�  "          @��\����@e��?\)���B�LͿ���?���{L�B��)                                    Bx��GR  �          @�녿�@W��7
=�(�B�\��?�����
=�x33C�                                    Bx��U�  �          @dz���R?�Q���R�1�RC\��R�����.�R�I�\C<Q�                                    Bx��d�  "          @K���\)?}p���L{C�H��\)��
=� ���a33CB�{                                    Bx��sD  �          @�G��(�@�
�j�H�n�HB�(��(��k���ff¢Q�CH��                                    Bx���  �          @a녿�R?�R�S33�C�f��R���R�G�{Cs\)                                    Bx�ᐐ  �          @@  �z���;���C@�׿z��
=����U�Cz�                                    Bx��6  
�          @��R��
=�������¥z�CZn��
=�   �U��R�
C�K�                                    Bx���  �          @�z�fff@#33�.�R�5G�B�{�fff?0���j�H��CǮ                                    Bx�ἂ  T          @x��>�Q�@%��7��?{B�=q>�Q�?#�
�s�
(�Bs�\                                    Bx���(  /          @�
=?xQ�@K��\)�  B���?xQ�?�{�s33��BY(�                                    Bx����  �          @��R��@HQ��(��{B��)��?�(��a��o��C{                                    Bx���t  
Z          @�����
=@l�Ϳ�Q����\B����
=@{�Z�H�E  C�                                    Bx���  "          @u���H@`  �W
=�M��B�����H@   �%�+�HB�\                                    Bx���  �          @XQ��@!G����
��B��H��?�Q����?Q�C�                                    Bx��f  �          @�Q���p�?���G��z�C&�
��p�����G��p�CAn                                    Bx��#  �          @�ff��ff?�=q���H�G�C&@ ��ff�����p��z�C?�                                    Bx��1�  T          @ᙚ��=q?�ff��  �p�C!Y���=q�B�\�����!  C<#�                                    Bx��@X  
�          @�Q���  ?�{�����!��C�)��  �^�R���H�.z�C=�)                                    Bx��N�  "          @���33?�p������C!E��33�aG�����&�C=�                                     Bx��]�  �          @��H��33?޸R��\)�${C E��33�p����
=�.�RC>��                                    Bx��lJ  T          @�  ��=q@33��  �33C����=q�
=q��{�0�C:p�                                    Bx��z�  �          @�G����H@!G��hQ��ffCW
���H>8Q���p��<C1u�                                    Bx�≖  �          @����XQ�@Dz��&ff��\)C	���XQ�?����u�;�C z�                                    Bx��<  �          @��H�z=q@�H�K��
��CE�z=q>�{�~�R�5�\C/�                                    Bx���  "          @�p��o\)@333�0  ��C=q�o\)?c�
�tz��333C&��                                    Bx�ⵈ  �          @���^�R@2�\�-p����
CG��^�R?h���q��9�C%k�                                    Bx���.  �          @�=q�	��@=p��p���G�B��
�	��?���\���W�RC��                                    Bx����  �          @��H�@7
=�z���=qCT{�?�ff�Q��K
=C�                                    Bx���z  �          @�Q��Tz�@�����ffCz��Tz�?W
=�<(��"�RC%�)                                    Bx���   �          @��H�c�
@'���
=��ffC���c�
?�
=�A���
C!�f                                    Bx����  �          @�33�`��@(�ÿ�z���(�C
=�`��?�(��&ff�	z�C33                                    Bx��l  T          @S33���\@(Q�?8Q�A]p�B�p����\@&ff�L���w�
B��)                                    Bx��  
Z          @l(�����@p�?��RA���B�zΌ��@*=q=�\)?��B��                                    Bx��*�  �          @n{�333?�\@J=qBk��B��H�333@L(�?�(�A��
B̞�                                    Bx��9^  �          @p  �L��?�(�@[�B�B�Býq�L��@E�@
=B	ffB��=                                    Bx��H  �          @r�\?E�?8Q�@g
=B�L�B,
=?E�@"�\@*=qB4  B�#�                                    Bx��V�  �          @j�H>aG�?z�H@aG�B�B���>aG�@.{@(�B'z�B���                                    Bx��eP  �          @l��>�ff?G�@a�B�33Bo�H>�ff@#33@#�
B2z�B��)                                    Bx��s�  �          @mp���G�?�ff@S�
B�(�Bӊ=��G�@E�?��HB{B�(�                                    Bx�゜  �          @�����(�@   �333��p�CG���(�?޸R��
=��ffCc�                                    Bx��B  
�          @�(����R@#33�,(���p�C.���R?5�h����C*��                                    Bx���  G          @�=q����@�\�7
=�ׅC������>����i���  C/��                                    Bx�㮎  �          @������R@  �<(���33C�����R>����l(��
=C0k�                                    Bx��4  �          @��
��(�@G��@  ��G�C(���(�>��R�p  �p�C0�
                                    Bx����  �          @�ff��p�@Q��A���z�C=q��p�>Ǯ�u�{C/��                                    Bx��ڀ  �          @�(���p�@���7
=��Q�C����p�?
=q�n{�z�C.{                                    Bx���&  �          @�\)���R@ff�O\)��C{���R=#�
�w
=�{C3�)                                    Bx����  �          @ə���p�@33�W
=� �RCp���p���\)�{��Q�C4��                                    Bx��r  �          @�33����@���
��  C�R����?(��5���33C,aH                                    Bx��  T          @�G���33@	��@6ffB9��B��f��33@U?�  A���B�                                     Bx��#�  T          @z�H�Ǯ@   @U�Bj�B�=q�Ǯ@^{?޸RA�  B��                                    Bx��2d  �          @��H��G�?��
@Y��BeffB��Ϳ�G�@Tz�?�A�G�Bݞ�                                    Bx��A
  �          @��
���@�@B�\BB�B�R���@c�
?���A�\)B޽q                                    Bx��O�  T          @�z�Ǯ@�
@N�RBM�B�G��Ǯ@^{?У�A�B�aH                                    Bx��^V  T          @�
=���\?���@p  B{Q�Cff���\@N�R@��B
�B�\                                    Bx��l�  "          @�(�����?�G�@}p�B�k�CY�����@K�@*�HB  B��                                    Bx��{�  �          @�p����?�\)@W�BYG�C�����@XQ�?���A�B�                                      Bx��H  �          @�=q��
=@�R@QG�BF
=B����
=@h��?���A�p�B�3                                    Bx���  �          @��
��
?��R@hQ�Ba�\C���
@?\)@��B�RB�=q                                    Bx�䧔  �          @����\@   @u�B\Q�C���\@o\)@
=qA�
=B�R                                    Bx��:  �          @�{��  @1G�@�BZQ�B�3��  @��R@��Aģ�B�8R                                    Bx����  �          @��
���@'
=@���Bc(�B�W
���@��@�RA�Q�B���                                    Bx��ӆ  �          @�p���@"�\@�=qBS(�C@ ��@��R@33A�(�B��                                    Bx���,  �          @�����@�
@��Bf��CW
���@�33@'�A�p�B�G�                                    Bx����  �          @���  @�\@���Bh��C	Ǯ�  @��H@C�
A��B�R                                    Bx���x  "          @�
=�5���
@���B�L�C5�H�5@H��@�  Be
=B�z�                                    Bx��  �          @��þ.{��\)@�Q�B���Cn���.{@G
=@�ffBq
=B�W
                                    Bx���  �          @�z῀  >L��@�
=B���C(�H��  @N�R@�\)BY��B�ff                                    Bx��+j  �          @�{�w�@�Q�    <�B��=�w�@��H�Mp����B��=                                    Bx��:  
Z          @ᙚ�qG�@��Ϳ+���{B�.�qG�@�(��mp����HC&f                                    Bx��H�  �          @��_\)@�p����
���B���_\)@�p���p��G�B��                                     Bx��W\  "          @����c33@������=qB����c33@����s33� �B��
                                    Bx��f  "          @����l(�@���u� ��B���l(�@�p��N{�㙚C Q�                                    Bx��t�  T          @����s�
@��\��
=�dz�B����s�
@���U����
CJ=                                    Bx��N  �          @����g
=@��>�
=@w
=B���g
=@�������\)C u�                                    Bx���  "          @ƸR�\)@���?uA{C���\)@�ff���y��C�                                    Bx�堚  �          @�{��@����R����C ����@�G��QG����C��                                    Bx��@  "          @�  ��\)@���+���{C�H��\)@w��N�R��RC(�                                    Bx���  �          @Ӆ���@�{������Cc����@��
�/\)�ď\C��                                    Bx��̌  �          @�33��ff@�녾B�\��Q�C����ff@��7
=���C.                                    Bx���2  �          @�z����
@�\)����33C=q���
@�z��0����G�C}q                                    Bx����  �          @������
@�  �\�R�\C:����
@tz��6ff��p�Cs3                                    Bx���~  �          @��
����@�\)�(�����C�����@N�R�3�
��p�CG�                                    Bx��$  �          @љ����R@�(����
�G�C0����R@<���Dz���=qC=q                                    Bx���  
�          @�(����@��ÿ��N=qC�����@)���Vff���C�                                     Bx��$p  �          @�
=��  @^{������\C�H��  ?�
=�\(���\C޸                                    Bx��3  �          @����ff@�ÿ�p���{C^���ff?xQ��<(��	\)C&�R                                    Bx��A�  �          @�33���@,(��=q��Cz����?�  �^{�z�C'��                                    Bx��Pb  �          @�33����@(���-p��ӅC����?L���l����C*��                                    Bx��_  "          @�  ���@{�\����C�����>����\)�3��C0��                                    Bx��m�  a          @����\)@p��Vff�	z�C�
��\)>��R��z��1=qC/�)                                    Bx��|T  �          @�33���\�\)�`����CP�{���\�p�׿��
����C^�R                                    Bx���  �          @�\)�]p������P���#�
CJ���]p��=p����R���C\��                                    Bx�晠  �          @�z��fff��ff�q��2CG���fff�G
=�!G���p�C\��                                    Bx��F  �          @ȣ����>�  ����=z�C0�f�����R��G�� ��CO�\                                    Bx���  �          @�
=��(�?�{��z��;p�C"�3��(���Q����
�:ffCF+�                                    Bx��Œ  �          @�\)����?�ff��=q�6�C#�3���׿��������4��CE                                    Bx���8  T          @�\)��@,���aG��
=C)��>�(�����9�RC.&f                                    Bx����  "          @����\)>8Q����
�>G�C1\)�\)�
=q�s�
���CPff                                    Bx���  
Z          @��W��\(���  �%Ca�
�W���p������33Cl�f                                    Bx�� *  �          @�(��hQ��7
=��(��-�
CZ:��hQ���  ������Ch��                                    Bx���  �          @�{�p  �z���G��1  CS�=�p  �����H����Cd��                                    Bx��v  �          @�ff�]p��ٙ������C�CN5��]p��qG��5����Ccz�                                    Bx��,  �          @����i�����������P�CDQ��i���`  �fff���C_�\                                    Bx��:�  �          @���g
=�   ��  �D�CP��g
=��ff�>�R��{CeY�                                    Bx��Ih  �          @ָR��������
=�@��CLs3������{�P  ��
=Ca��                                    Bx��X  �          @�\)��=q��\)��z��O�CH����=q��\)��z����Cb\                                    Bx��f�  �          @��H��
=�����z��Y{CE���
=���H�����Q�Ca                                    Bx��uZ  �          @������Ϳ����
=�S  CA��������\����Q�C^Ǯ                                    Bx��   �          @����=q�������@p�C:����=q�Tz�����Q�CV�{                                    Bx�璦  �          @����R�����z��Bz�C;=q���R�Vff������CWh�                                    Bx��L  �          @����녾�  �����>z�C6�����?\)����G�CS޸                                    Bx���  T          @�\)���þ�������*��C9������>�R�z�H�ffCQu�                                    Bx�羘  �          @�
=���H�#�
���
�4z�C4}q���H�.�R���\�G�CP0�                                    Bx���>  �          @�\)��  =�\)���C(�C35���  �333����� �CR��                                    Bx����  �          @�\���<������G=qC3�f����4z�����"�CS��                                    Bx���  �          @��H���
>.{����D�HC1�3���
�+������$�CR
                                    Bx���0  �          @�R��z�>\)��Q��Gp�C2B���z��1G���Q��%ffCR�)                                    Bx���  �          @�p���?
=q��(��C(�C-^����
=���
�+ffCN�=                                    Bx��|  �          @�
=���?.{�����I{C+� ��������\�3
=CO
                                    Bx��%"  �          @������?�\)�����?
=C&�=���ÿ����{�5=qCI�                                     Bx��3�  �          @�33�mp�>���ʏ\�nffC1�3�mp��L(����R�@��C\��                                    Bx��Bn  �          @��
�n{<#�
��33�nz�C3�\�n{�Tz���p��=ffC]��                                    Bx��Q  �          @�(����\>����z��a�C1�3���\�E�����8{CY.                                    Bx��_�  �          @�(��{��#�
��\)�g  C40��{��Q������8
=C[��                                    Bx��n`  �          @�{�q녿\(��ə��i�\C@�
�q���������&��Cbٚ                                    Bx��}  �          @��H����?fff�����PffC(z������  ��z��=��CO!H                                    Bx�苬  �          @�=q��G�?0����ff�L��C+\)��G������
=�5�CO��                                    Bx��R  �          @�33��Q�?������O�C-@ ��Q��#�
��ff�433CQ�
                                    Bx���  �          @��\)>Ǯ�Å�b�
C.u��\)�7
=����?{CW��                                    Bx�跞  �          @�p���  �����(��_�HCC5���  ��(�����33Ca�                                    Bx���D  �          @�R�QG������Ӆ�y��CF��QG����
���\�+�
Ci#�                                    Bx����  �          @�Q��^�R�}p��ҏ\�tCC��^�R�������H�*�Cf��                                    Bx���  �          @�G��R�\��p���p��xCH� �R�\���������(�Ci��                                    Bx���6  �          @�ff�333�:�H���H{CB���333��{���R�=
=Cl:�                                    Bx�� �  �          @�(��S�
=��
�ҏ\�}(�C2�
�S�
�W
=����H��Cak�                                    Bx���  �          @����y��?�p����R�\{C@ �y��������V�\CMG�                                    Bx��(  �          @�
=�y��@����z��P�RCxR�y�����\�ȣ��e(�CB�R                                    Bx��,�  �          @�
=��z�@   ��  �=CO\��z�8Q���  �U��C=\)                                    Bx��;t  �          @�����G�@-p���(��:�RC� ��G�����Q��Y{C:n                                    Bx��J  �          @�G��w�?�\)�����Y�
C33�w����
���
�_
=CI�=                                    Bx��X�  �          @�\)�W
=>\)��p��|�HC1��W
=�U�����JQ�C`�)                                    Bx��gf  �          @陚�\�Ϳ!G���(��tC>h��\���vff���
�2�Cd&f                                    Bx��v  �          @���hQ��G��ҏ\�tp�C5���hQ��`  ��=q�?\)C_�q                                    Bx�鄲  �          @���qG�>\)����q�C1�)�qG��U������C��C]xR                                    Bx��X  �          @�����33>�ff��{�_z�C.
��33�<����  �>=qCV33                                    Bx���  �          @�z���Q�?(����{�Z��C+���Q��1G���33�?�\CS�
                                    Bx�鰤  �          @�\)����?
=��G��\�C,�������8Q������>�CTk�                                    Bx��J  �          @�{����?&ff��z��c(�C+ff�����8Q������EQ�CU��                                    Bx����  �          A (���(�>�z���{�c  C0:���(��Mp����
�={CXB�                                    Bx��ܖ  T          A���\)��\�ָR�ZC:0���\)�z=q��\)�'33C[��                                    Bx���<  �          A���녿�\)�љ��S�HCA�������H�����ffC^�                                    Bx����  �          A��zῆff�ڏ\�\�\C@���z���p�����z�C_�H                                    Bx���  �          A��w�>�{����z�RC/�w��e���
�M�RC^�
                                    Bx��.  �          A��p��?W
=��=q�
=C'p��p���S�
��(��[��C]Y�                                    Bx��%�  �          A{�n{?#�
� ����C*5��n{�e��
=�Y�\C_�                                    Bx��4z  �          A(��o\)?.{��\�C)�f�o\)�g���33�Z�C`�                                    Bx��C   �          A  �tz�?(������C*@ �tz��g
=�陚�Y  C_k�                                    Bx��Q�  �          A	���r�\?�p����v  C���r�\�%����e�HCVB�                                    Bx��`l  T          A����
@�{�p�����C����
@#33��z��C��                                    Bx��o  �          A �����@���:�H��\)C	�����@�Q��Y����Q�C+�                                    Bx��}�  �          A ����G�@�{�Y����G�Cٚ��G�@s�
�P����\)C��                                    Bx��^  T          AG���z�@�G�    ���
CE��z�@�
=�>{��ffC
��                                    Bx��  �          AG���@�{=u>�ffC���@���'
=���RC��                                    Bx�ꩪ  �          A����@�Q쾣�
�\)C)��@�  �Vff��33C	^�                                    Bx��P  �          A���G�@�ff�#�
����Cn��G�@�=q�C33��ffC8R                                    Bx����  �          A����33@��H<#�
=uCG���33@���6ff��
=C��                                    Bx��՜  �          A (���G�@�Q�>8Q�?�G�C
����G�@��
�"�\��C��                                    Bx���B  �          @�����@��Ϳ+����HC�{���@�Q��]p���p�C5�                                    Bx����  �          @�{��=q@�p���ff�3�
CǮ��=q@xQ���33���HC.                                    Bx���  �          @�����(�@����G��N{C	
=��(�@g
=��{� �CL�                                    Bx��4  �          @�����
@�����
=�33C����
@�p��u��C0�                                    Bx���  �          @������@�녿z�����C����@��R�]p��ԏ\C	�                                    Bx��-�  �          @�������@\?s33@�p�B�L�����@�
=�G���z�C W
                                    Bx��<&  �          @�=q��ff@�G�?��HAQ�B�����ff@�=q�˅�C
=B�L�                                    Bx��J�  �          @���(�@�@$z�A���B�{��(�@�ff�+�����B�aH                                    Bx��Yr  �          @�
=����@�p�@\)A�(�B�������@��׿s33��=qB���                                    Bx��h  �          @�  ��
=@�\)>Ǯ@A�Cu���
=@�z��$z���C�\                                    Bx��v�  �          @�(����@��H�������B�\���@��R�E��{C!H                                    Bx��d  �          @�G����H@�{�aG���{B�.���H@�p��p�����C5�                                    Bx��
  �          @�33�k�@�����\��
=B�Ǯ�k�@������#�\Cs3                                    Bx�뢰  �          @�33��33@�  �����v�\B��f��33@~�R���H�(�C�{                                    Bx��V  T          @��H�S33@-p���33�Wz�C�H�S33��R��p��yz�C>��                                    Bx���  �          @��H�]p�@N{�����Fp�C	��]p����
��p��v��C4h�                                    Bx��΢  �          @�p��}p�@mp������+C�{�}p�?(����\)�d�C*�{                                    Bx���H  �          @��
�mp�@b�\����8
=CQ��mp�>\���
�n�RC.&f                                    Bx����  T          @�G��x��@�  ��  �{C=q�x��?�����X�C!Y�                                    Bx����  �          @��H��=q@u��33�Q�C����=q?�����{�U��C$xR                                    Bx��	:  �          @�G���ff@w���G����C	^���ff?�����
�VG�C%ٚ                                    Bx���  �          @���33@��p�����C���33?�Q���\)�>��C(�                                    Bx��&�  �          @����R@�  �tz����\C�=���R@(���{�H��C��                                    Bx��5,  �          @�Q���33@������\�{C
�=��33?��������I33C$�                                    Bx��C�  �          @�����H@~{������C#����H?����{�H(�C$G�                                    Bx��Rx  �          @�z�����@�R���
G�Cu����;.{�����C5��                                    Bx��a  �          @�����G�@���{��\C�3��G���  ��\)�-{C6��                                    Bx��o�  �          @�p����@-p���=q�ffCJ=���=u����7\)C3T{                                    Bx��~j  �          @����@*�H����� �C�{������
��\)�>C4�f                                    Bx��  �          @�R����@�����,��C�����׿@  ���R�<Q�C<xR                                    Bx�웶  �          @������?�z���G��+��C")���Ϳ�����ff�2�\C?                                    Bx��\  �          @�(���{?��
���H�.Q�C&0���{��
=�����,�\CCff                                    Bx��  �          @�{�y��@P  ��G��-�C33�y��>�G������_  C-�)                                    Bx��Ǩ  �          @���(�@�������-(�B�𤿼(�?����\)��B���                                    Bx���N  �          @��Ϳ�\)@����
=�z�B����\)@&ff��33�}  B�z�                                    Bx����  �          @ۅ��@�  ��(���B�W
��@Q����p�B̽q                                    Bx���  �          @�=q�.{@��R��33�*=qBî�.{@G��θRaHB�L�                                    Bx��@  �          @ۅ�Y��@�p����,�RBǮ�Y��?�����Q���B�L�                                    Bx���  �          @�(��:�H@�ff������RB�G��:�H@\)��(�.BԮ                                    Bx���  �          @�R�@  @�33��z��Q�B�@  @A��Ϯ��Bϣ�                                    Bx��.2  �          @�Q�u@���mp����
BŸR�u@aG��ȣ��o(�BҀ                                     Bx��<�  �          @�R���\@����i����p�B�{���\@p�����H�i��B�Q�                                    Bx��K~  �          @�(��G�@ə��qG����B���G�@g
=��z��o�
B�Q�                                    Bx��Z$  �          @�=��
@У��@����=qB���=��
@�{���H�YQ�B���                                    Bx��h�  �          @�\)>��R@�ff�'
=��ffB��3>��R@�G����\�K�B�.                                    Bx��wp  T          @���Q�@�\)�����B��\��Q�@�������@��B�33                                    Bx��  �          @�  >�{@��
�{��z�B�(�>�{@�p���{�?�B��R                                    Bx�피  �          @ٙ�>��@�ff����33B�u�>��@��\��Q��>=qB���                                    Bx���b  �          @�G�?�@�p�������B��R?�@�=q�i���	ffB���                                    Bx���  �          @ə�?   @�\)=L��>�G�B���?   @�ff�@����RB��                                     Bx����  �          @��Ϳ   @��������RB��H�   @\)���E(�B�=q                                    Bx���T  �          @���Q�@��\��=q���B�zᾸQ�@@  �`  �D�B���                                    Bx����  �          @��R����@��
�<(����\B�\����@���ff�cp�CaH                                    Bx���  �          @�z��&ff@���a���B�L��&ff@��33�cffC
                                      Bx���F  �          @��
���@�\)�c33�ffB�����@������j��C\)                                    Bx��	�  �          @ʏ\���@��
�qG��=qB����@=q��p�p�B�p�                                    Bx���  �          @ʏ\���H@��
�qG���B�\)���H@�H����~  B�8R                                    Bx��'8  T          @����R@��
�e���B�uÿ��R@.�R��(��tG�B�L�                                    Bx��5�  T          @�\)����@��R�`  �
=B�#׿���@6ff��33�n��B��                                    Bx��D�  �          @�Q���@����9���ӅB��Ϳ��@\(���\)�U�RB�                                    Bx��S*  �          @љ���
=@���8Q���z�B�Q��
=@]p����R�S  B�L�                                    Bx��a�  �          @���ff@�  �7���  B����ff@Z�H���P�B�(�                                    Bx��pv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��   V          @�G���(�@���8����B�\��(�@`������U33B�(�                                   Bx���  �          @Ӆ���R@�G��@�����HBۀ ���R@X������V
=B�Ǯ                                   Bx��h  �          @�z��z�@���R�\��(�B�B��z�@G
=��
=�^�B�ff                                   Bx��  �          @�p����
@����h���(�B�(����
@8����  �m�B�W
                                    Bx�  �          @�  �@�p��[���\B�=q�@E����a�B�(�                                    Bx���Z  �          @Ӆ��\@�=q�J�H���B���\@HQ����H�X{B�=q                                    Bx���   
�          @�33>#�
@A�@l��BJ\)B�Ǯ>#�
@�{?��
A���B��q                                    Bx���  
�          @��H����@s�
@ffA��B������@�
=>k�@>�RB��=                                    Bx���L  �          @��Ϳ�R@��H>���@2�\B�����R@���*=q���B��                                    Bx���  �          @��\�!G�@�?uA�
B�G��!G�@�
=��G���B���                                    Bx���  �          @��ÿ^�R@���?�\)A[33B���^�R@�Q쿗
=�f�RB��f                                    Bx�� >  �          @��
��ff@�ff?\)@�
=B�Ǯ��ff@��H�   ��=qB�u�                                    Bx��.�  �          @�z῰��@�
=������B�녿���@�\)�8Q���z�B�u�                                    Bx��=�  �          @�  ���@�(���G���{B�𤿋�@��*=q��  B�G�                                    Bx��L0  �          @n{�xQ�@_\)��\)���B��f�xQ�@<(���z���p�B�aH                                    Bx��Z�  �          ?��
��?^�R�=p���HB����>�녿���hG�C�                                    Bx��i|  �          @   �n{?�=q���
�M=qC�H�n{=������C-�
                                    Bx��x"  �          @;����?333�#�
�~�RC�\�������%��{CQ�3                                    Bx���  �          @4z῕@���\)����B�׿�?�
=�(��SffC�3                                    Bx��n  T          @W
=���R?u�0  �d��Cff���R���7��uCE0�                                    Bx��  �          @���ff@H�ÿ�����B���ff@���(����
C	��                                    Bx�ﲺ  T          @�(��=q@Vff���R��G�B�p��=q@��+��z�C��                                    Bx���`  �          @�33�@HQ����
�V�RB�p��>�{���{C*�\                                    Bx���  �          @����G�@]p������0��B���G�?�33����C.                                    Bx��ެ  �          @�Q��HQ�@tz���=q�+=qCT{�HQ�?��H���\�pffCٚ                                    Bx���R  �          @�p��dz�@q���p��-�CW
�dz�?��\���
�k�C${                                    Bx����  �          @�
=����@QG����
�8�C�
����>�Q������e(�C.�f                                    Bx��
�  �          @�Q����@P�����/33C�)���>�G���33�Z��C.!H                                    Bx��D  �          @ə��/\)��\)���H�c��CVL��/\)��G��g���Ck�)                                    Bx��'�  T          @�(����J�H���
�>Q�CmaH������	����\)Cv�=                                    Bx��6�  �          @ȣ׿���=q���B�CW�f���y����Q��9{Ct�R                                    Bx��E6  �          @�G��Ǯ�L����G���C;:�Ǯ�@  ��  �dCr�=                                    Bx��S�  �          @�{����=#�
��Q��RC2(������1G����\�o��Ct�                                    Bx��b�  �          @Ӆ�,�ͿL����33CD���,���^�R���
�?�Ch.                                    Bx��q(  �          @�\)�0�׿.{���HCA�)�0���\(���=q�C��CgO\                                    Bx���  �          @����QG��
=��  �
=C>=q�QG��j�H��ff�D�
Cd=q                                    Bx���t  �          @�33�E������{��C:��E�aG�����P  Cd��                                    Bx��  �          @Ϯ��H��\)��Q��C5����H�6ff��G��Z�HCe�3                                    Bx���  �          @��ÿ�>\)��\)�C/{���z���Q��h
=Cg�                                    Bx��f  �          @�\)����=L�����R� C2Y����������ff�f  CiJ=                                    Bx���  �          @���\�L�������C;uÿ\����G��]��Cl�                                    Bx��ײ  �          @����
=>��R�����fC'����
=��
=���
�t�Ci��                                    Bx���X  �          @��
��?�p����Hk�C�{�����
��z�8RCS��                                    Bx����  T          @����ٙ�@G�����lp�B����ٙ��.{��Q���C9�                                    Bx���  T          @��\�   ?������z(�C
#��   �J=q������CI��                                    Bx��J  �          @�p���
?���
=u�CE��
��z�����{CN�H                                    Bx�� �  �          @����
@�\�����HC
}q��
�����G�ffCM5�                                    Bx��/�  �          @���(�?��
���H��C�)�(���=q��ffaHCPz�                                    Bx��><  �          @����ff?�\)��Q��w�\C
^��ff�@  ��G�\CG�R                                    Bx��L�  �          @��Ϳ�\@����(��ffB�G���\@,(����H�B��q                                    Bx��[�  �          @�\)��H@����33�,  B虚��H@�����  C�                                    Bx��j.  �          @�=q�333@��H��33�8��B����333?��
��\z�Cp�                                    Bx��x�  �          @�׿޸R@������{Bٙ��޸R@!���Q��\)B��                                    Bx��z  �          @�=q��@���y���
\)B�����@8������j33B�p�                                    Bx��   �          @�녾�@أ�?˅AR�RB�Ǯ��@�  ���]p�B���                                    Bx���  �          @�׾�{@�  @(Q�A�=qB��쾮{@�R�8Q���
=B�p�                                    Bx��l  �          @�{����@ٙ�?�  A&�RB�zᾨ��@�(���(���Q�B���                                    Bx���  �          @�(���@�\���k�B����@Å�g���\B��f                                    Bx��и  �          @�=q���@ٙ�����Y�B��\���@�=q��G��!�HB�G�                                    Bx���^  �          @�=q<�@У׿���5��B�k�<�@�
=���
��B�G�                                    Bx���  �          @Å?   @��?�@��HB��?   @����R��G�B��)                                    Bx����  �          @��@ff@��H@@��A��B�B�@ff@�\)?:�H@��HB�(�                                    Bx��P  �          @�(�@Q�@�p�@W�B\)Bq33@Q�@��?���A2=qB�33                                    Bx���  �          @Ϯ@Fff@O\)@���B8{B9(�@Fff@��@,(�A��Bh�                                    Bx��(�  T          @׮@b�\?�G�@���B_p�A�ff@b�\@g�@�z�B#�RB6G�                                    Bx��7B  �          @ڏ\@^{?�\)@�Q�Bf��A��H@^{@e�@���B,  B7��                                    Bx��E�  �          @��@`  @\)@�=qBG  B��@`  @��@Z�HA�BPff                                    Bx��T�  �          @љ�@k�@<(�@�ffB-z�B@k�@��
@*=qA�{BN
=                                    Bx��c4  �          @ҏ\@vff@7
=@�\)B,G�B�@vff@�=q@.�RAģ�BG�\                                    Bx��q�  �          @��H@g
=?��@��BV��A��H@g
=@N{@��B!ffB&��                                    Bx��  �          @��@n�R?��@���BT�A{@n�R@1G�@�\)B)�B�                                    Bx��&  �          @�=q@U@�\@��BMQ�B	p�@U@���@`��B\)BPz�                                    Bx���  �          @ҏ\@dz�?�p�@�(�BN{A�(�@dz�@���@n{B
33BB��                                    Bx��r  �          @�33@`��@
=q@�z�BMQ�A��H@`��@�
=@i��B��BI                                      Bx��  �          @Ӆ@Mp�@&ff@�z�BL�HB�@Mp�@��@]p�A���B\��                                    Bx��ɾ  �          @�p�@AG�@0  @�Q�BP��B)G�@AG�@�G�@`  A��Bg�                                    Bx���d  �          @�p�@@  @.{@���BRz�B)  @@  @���@c33B �Bg                                    Bx���
  �          @Ӆ@Q�@?\)@���B<33B)33@Q�@���@=p�A�(�B^33                                    Bx����  �          @�Q�@J�H@y��@��B�BK\)@J�H@��?�(�A�p�Bl�\                                    Bx��V  �          @�G�@vff@H��@���B=qB��@vff@��H@\)A��BH                                      Bx���  �          @�G�@~{@�R@�{B7ffA���@~{@�=q@N�RA�
=B6�
                                    Bx��!�  �          @��@|��@ff@�z�B4��A���@|��@���@H��A噚B9ff                                    Bx��0H  �          @�=q@��\@
=q@�(�B4G�A�G�@��\@~{@Mp�A�B0�                                    Bx��>�  T          @�Q�@�33@\)@��B/p�A���@�33@~�R@C�
A�33B0z�                                    Bx��M�  �          @�  @vff@#33@���B1�RB  @vff@���@<��A�B?�H                                    Bx��\:  �          @љ�@g�@*=q@�Q�B:�B33@g�@�
=@G�A�  BL                                      Bx��j�  �          @�ff@O\)@0  @��BCG�B!
=@O\)@��H@K�A�B[                                      Bx��y�  �          @ָR@�33?�ff@�(�BH  A��H@�33@Z=q@���B�B                                    Bx��,  �          @�@r�\@   @�(�BH��A�ff@r�\@���@p  B��B;
=                                    Bx���  �          @�{@x��?��@�=qBSz�Ap��@x��@P  @��HB"G�B�                                    Bx��x  �          @�\)@��H?��@�ffB=��A�p�@��H@Vff@xQ�B{B                                    Bx��  �          @���@��@5�@;�A��HA�@��@tz�?�AMBz�                                    Bx����  �          @��H@��@U@"�\A��BQ�@��@��?Y��@�RB
=                                    Bx���j  �          @�ff@��R@A�@.�RA���A�@��R@z=q?�33A z�B��                                    Bx���  �          @�
=@�\)@p��@N�RA�G�B��@�\)@���?�ffA2�HB;z�                                    Bx���  �          @ָR@�=q@j=q@7�A���B
=@�=q@�G�?�G�A\)B-G�                                    Bx���\  �          @�
=@��@qG�@=qA�Q�BQ�@��@�{?��@�\)B&��                                    Bx��  �          @�\)@���@�
=@Q�A�p�B%�R@���@�\)=�G�?n{B2��                                    Bx���  �          @��@���@�@
=A���B>ff@���@��>�?�{BK=q                                    Bx��)N  �          @�\)@�@��@0��A��
B<33@�@��?z�@��BM�                                    Bx��7�  �          @ٙ�@~�R@��@6ffAř�BG�H@~�R@�  ?��@�(�BXp�                                    Bx��F�  �          @أ�@}p�@��@��A��
BLG�@}p�@�\)=�?�ffBX�\                                    Bx��U@  �          @أ�@��
@�=q@A�{BE��@��
@��=u?��BQ��                                    Bx��c�  �          @�  @��H@�  @�A��B>ff@��H@�{�����BH\)                                    Bx��r�  �          @�{@P��@�Q�@   A�  Bh�H@P��@��\<�>��Bs{                                    Bx��2  �          @׮@h��@�
=@A��B\�@h��@�����
�1�Bd33                                    Bx���  �          @�=q@s33@��\?�G�Ap��BZ  @s33@�=q�(�����B^��                                    Bx���~  �          @ٙ�@tz�@��\?Tz�@��B^ff@tz�@�������YB[
=                                    Bx���$  �          @У�@z�H@���?
=q@�\)BS
=@z�H@��Ϳ�Q��q�BMff                                    Bx����  T          @���@��@�?�@�(�BE  @��@�{�����b�HB?ff                                    Bx���p  �          @�  @��
@���?�@���B>
=@��
@�=q���R�T  B9{                                    Bx���  �          @��H@�  @�
=?=p�@θRBE�H@�  @��\���F�HBBp�                                    Bx���  �          @��
@��@��?�  A
�\BJ33@��@�Q쿜(��)�BI�                                    Bx���b  �          @ҏ\@�  @��R?k�A z�B9Q�@�  @�������B8�                                    Bx��  �          @�p�@�G�@w�?+�@���B{@�G�@r�\��G��p�B��                                    Bx���  �          @�=q@��@n{>�G�@�G�BG�@��@dzΐ33�(z�B�                                    Bx��"T  �          @ʏ\@�  @a녽�G���  B�R@�  @J=q�����f=qA��\                                    Bx��0�  �          @ə�@��@^�R�aG���B=q@��@Dz��33�s33A���                                    Bx��?�  �          @�p�@�
=@mp������,(�B�@�
=@P  �������A�\)                                    Bx��NF  �          @θR@�33@fff�Ǯ�_\)B�\@�33@G
=������
=A��                                    Bx��\�  �          @��@1G�@���>��@G�B��
@1G�@���G���33B{                                    Bx��k�  �          @�z�@'
=@��R?�G�A��B���@'
=@��H��ff�YB�Ǯ                                    Bx��z8  T          @�{?��@��?��
Av�RB�  ?��@�\)�}p���B���                                    Bx����  �          @ָR?W
=@θR?ǮAX(�B�8R?W
=@��ÿ��R�*ffB�aH                                    Bx����  �          @�p�?�p�@�=q?�p�A*�\B�k�?�p�@ȣ׿�p��L��B�.                                    Bx���*  �          @�G�@G�@�Q�?�
=AL  B��)@G�@�=q��z��$��B�.                                    Bx����  �          @У�?˅@�{@��A�33B���?˅@�G����H��=qB��{                                    Bx���v  �          @���?��@�=q?�p�A��B�{?��@�33�.{��\)B��                                    Bx���  �          @Ӆ?z�H@ʏ\?��@�p�B�ff?z�H@�Q�����B�u�                                    Bx����  �          @�\)?�  @���?�G�AX��B��\?�  @��
��=q�{B�                                    Bx���h  �          @�z�@�@��?�  AYB��q@�@��R���\�
=B�\)                                    Bx���  �          @�G�?���@��?���A��B�B�?���@��H��
=�RffB���                                    Bx���  �          @�=q@�\@�ff�aG���p�B�(�@�\@��H�'��ď\B�B�                                    Bx��Z  �          @��?��@���=�?�\)B�p�?��@�����ffB�#�                                    Bx��*   �          @�33?��@�  ��33�N{B�(�?��@��H�0  �υB�.                                    Bx��8�  �          @�p�@�
@�{��=q��
B���@�
@���X��� \)B�{                                    Bx��GL  �          @�ff@	��@�����
����B�
=@	��@�G��|����
B�\)                                    Bx��U�  �          @��@mp�@�?�(�AV{BT33@mp�@�33�#�
���\BX                                      Bx��d�  �          @�  @��@�@Dz�A؏\B0��@��@��?�\)ABFp�                                    Bx��s>  �          @��H@p  @��@;�A�\)BJ
=@p  @�33?\(�@�RB[                                    Bx����  �          @љ�@Z�H@���@N{A�{BTG�@Z�H@��R?�\)AffBg                                    Bx����  �          @�33@^�R@��@\��A��RBO
=@^�R@�p�?�33AC�
Beff                                    Bx���0  �          @�ff@x��@��@��A��BN�\@x��@��R>u@�BZG�                                    Bx����  �          @���@@  @�ff@7�A�{Bp
=@@  @��?��@�  B|Q�                                    Bx���|  �          @��H@1G�@�  @A�z�B}G�@1G�@�\)�L�;��B�\)                                    Bx���"  T          @�G�@(�@�(�@��A��B�.@(�@��׾�\)�(�B��f                                    Bx����  �          @Ϯ@ff@��@	��A���B��f@ff@�����=q���B��=                                    Bx���n  �          @�
=@�@��@ffA�G�B�W
@�@\��\)�#�
B�B�                                    Bx���  �          @�ff@33@���?�A���B�
=@33@��H����
=B��q                                    Bx���  �          @Ϯ?�p�@�{?�
=A���B��=?�p�@�
=����=qB��                                    Bx��`  �          @���@0��@�z�@\)A�ffB{��@0��@��H��\)��RB�k�                                    Bx��#  �          @�
=@`  @���@'�A��BV�R@`  @��?�@�ffBdQ�                                    Bx��1�  �          @ə�@s�
@z=q@G�A��B6�@s�
@�=q?�=qAC
=BN��                                    Bx��@R  �          @�{@Dz�@�z�@5A�G�Bgz�@Dz�@��?0��@���Bu=q                                    Bx��N�  �          @�(�@�
@�33@%A��B�B�@�
@�{>�\)@#33B�p�                                    Bx��]�  �          @�ff@6ff@���@ffA��Bv�\@6ff@���=���?\(�B                                      Bx��lD  �          @˅@>{@�=q@=qA�=qBn�R@>{@��>k�@
=Bx�                                    Bx��z�  �          @�(�@AG�@���?�=qA�Bq
=@AG�@��\�\�Z=qBvQ�                                    Bx����  T          @˅@5�@�z�?�p�A}�ByQ�@5�@�z�   ��=qB}z�                                    Bx���6  �          @�p�?�33@���?��HA���B���?�33@��H��G��|(�B�Q�                                    Bx����  �          @ȣ�?�Q�@�p�?�{A"�HB�#�?�Q�@�(����
�<��B���                                    Bx����  �          @�(�?��@��H?=p�@�\)B���?��@���=q�o33B�Ǯ                                    Bx���(  �          @���?��@��?��
AC�B�W
?��@�G������B��=                                    Bx����  �          @�����  @��R?��A�p�B�
=��  @�������Q�B���                                    Bx���t  �          @�������@��?�Q�A6=qB�  ����@����
=�4Q�B�                                      Bx���  �          @��H>Ǯ@���?   @�\)B���>Ǯ@�  ��������B�L�                                    Bx����  �          @��?^�R@�\)�����B�u�?^�R@��R����{B��                                    Bx��f  �          @�(�?���@�{@��A�{B�u�?���@�33�\)��ffB��R                                    Bx��  �          @��
?��H@�=q@�A��B�G�?��H@��=u?
=qB�(�                                    Bx��*�  �          @��H?�@�ff@�\A��\B�8R?�@�����z��(Q�B�(�                                    Bx��9X  �          @���@   @�G�@
�HA�  B�8R@   @�ff���Ϳp��B�                                    Bx��G�  T          @��?���@��@�A�\)B���?���@���.{����B��                                    Bx��V�  �          @��
?�G�@�ff?���ARffB��?�G�@����k����B���                                    Bx��eJ  �          @�G�?��@�p�?�  AG�B��?��@��H��z��H  B��                                    Bx��s�  �          @�  ?�{@��?^�R@�\)B�\)?�{@��ÿ\�Y�B��                                    Bx����  �          @��
?���@Å?\(�@���B��?���@�\)��G��[�B��=                                    Bx���<  �          @ə�?�33@��?:�H@�B��)?�33@�\)����qB�B�                                    Bx����  �          @�  ?n{@�=q?���A!�B���?n{@�G����\�<(�B��                                    Bx����  �          @ƸR?���@�{?�  A:{B��H?���@�
=�����33B�                                    Bx���.  �          @ƸR?�ff@�ff?�z�A,  B�L�?�ff@�ff��z��,��B�L�                                    Bx����  �          @�G�?��@�=q?xQ�AffB��\?��@����\)�I�B�B�                                    Bx���z  �          @�G�?���@�=q?n{A��B��f?���@�\)��33�NffB��                                    Bx���   �          @�  ?�{@�  ?���A"{B�p�?�{@�
=��(��4Q�B�W
                                    Bx����  �          @�\)?n{@���?���A1G�B�ff?n{@�������(��B�k�                                    Bx��l  �          @�G�?(��@��?�
=A4��B�{?(��@�(�����&�\B��                                    Bx��  �          @�=q>aG�@�Q�
=���\B��q>aG�@���1�����B�8R                                    Bx��#�  �          @�z�<#�
@�녿��
�PQ�B��q<#�
@�{�P  ��B��3                                    Bx��2^  �          @�=q@(Q�@��H?��HA�
=Bz�@(Q�@��R��\)��RB���                                    Bx��A  �          @�  @=q@��?�33AW�B�33@=q@�ff�(����33B�Q�                                    Bx��O�  �          @���@��@��?��RA>=qB�p�@��@�녿^�R��B��                                    Bx��^P  �          @���?��H@�p�?��@��B��?��H@�
=��=q�r�\B���                                    Bx��l�  �          @���?��\@��>.{?�=qB���?��\@�Q��   ��B�#�                                    Bx��{�  �          @��?=p�@�{�#�
��
=B�Ǯ?=p�@�Q��{��{B��3                                    Bx���B  �          @��R>�@�33�k��
=B�>�@�33�@  ��RB��3                                    Bx����  �          @��?�@��H�#�
��Q�B�p�?�@��������B���                                    Bx����  �          @�z�?�{@���>L��?���B�33?�{@�{����ffB��                                    Bx���4  �          @\?��\@�����  B��
?��\@�Q������HB�{                                    Bx����  
�          @�{?�\)@�
=������B��?�\)@�Q���
��
=B�B�                                    Bx��Ӏ  �          @���?�=q@��׾����HB�k�?�=q@�ff�"�\��ffB�B�                                    Bx���&  �          @�\)?��
@�G��
=q��p�B�
=?��
@�ff�&ff���HB�Q�                                    Bx����  �          @�
=��@��R@p�A��B�B���@��R>�ff@��RB�k�                                    Bx���r  �          @�ff��ff@�(�>�z�@;�B���ff@�(������HBڽq                                    Bx��  �          @�
=��
=@��������
B��Ϳ�
=@�{�z�����Bم                                    Bx���  �          @�{?Q�@��?��HA��RB�G�?Q�@�33��
=����B���                                    Bx��+d  �          @�G�?�ff@��@�A�{B�
=?�ff@�p�>�Q�@fffB�\                                    Bx��:
  �          @��H?�\)@�\)?��Aip�B���?�\)@��;��H��z�B�.                                    Bx��H�  �          @�G�@%@���?��\A*{Bv33@%@��\�B�\���
Bw(�                                    Bx��WV  �          @�@@��@�?�33Ar�\BY  @@��@��;B�\�G�B^��                                    Bx��e�  �          @�{@$z�@s33?�  Atz�B_��@$z�@�  �8Q��p�Bd��                                    Bx��t�  �          @��\?!G�@�녿���r{B�?!G�@����;��
=B�(�                                    Bx���H  
�          @��ýL��@�z�=p��	B�{�L��@������33B�8R                                    Bx����  �          @�=q>��R@��ÿ�(��c�
B��>��R@�G��5��B�(�                                    Bx����  �          @�
=�#�
@�33��
=��G�B�� �#�
@��H�g
=�%z�B�\)                                    Bx���:  �          @�z�?�@�G���G����\B�33?�@��L(��\)B���                                    Bx����  �          @�G�?\)@�����z�HB��?\)@��
�B�\���B�k�                                    Bx��̆  �          @�z�\)@��׿�
=��{B�(��\)@�  �j=q�!�
B�                                    Bx���,  �          @�=q=�\)@�녿����
=B�u�=�\)@�z��X���B�#�                                    Bx����  �          @�(�>�z�@��ÿ���  B���>�z�@����i���!�B��                                    Bx���x  �          @��>B�\@������33B�Ǯ>B�\@�=q�vff�-p�B���                                    Bx��  �          @�G����
@�33������B�� ���
@����q��,�RB���                                    Bx���  �          @�{�E�@��Tz���B�녿E�@���%�����B�(�                                    Bx��$j  �          @��׿&ff@�Q쿱��v{B���&ff@�
=�C33�(�BŔ{                                    Bx��3  �          @�G���@�p���=q���\B�aH��@~{�Z�H�"  B�G�                                    Bx��A�  �          @�{����@�G���p���\)B������@�G��g
=�&�B�33                                    Bx��P\  �          @���>8Q�@�(�� �����B���>8Q�@x���e��*z�B��R                                    Bx��_  �          @�(�?W
=@��\����B�W
?W
=@dz��dz��1{B���                                    Bx��m�  �          @�{?h��@��������B��
?h��@j=q�c�
�-G�B�                                      Bx��|N  �          @�\)?p��@��
�\)����B���?p��@dz��l(��3��B�u�                                    Bx����  �          @��?W
=@�(�������B��?W
=@h���c33�.33B�\                                    Bx����  �          @�33?Q�@��H�   ���B��R?Q�@w
=�b�\�'ffB���                                    Bx���@  �          @�ff?0��@�\)��p�����B�L�?0��@qG��^�R�(��B��                                    Bx����  �          @��<#�
@�Q��ff���
B��R<#�
@j�H�u��8�
B���                                    Bx��Ō  �          @���.{@��
�z���B��.{@r�\�u�5��B�\                                    Bx���2  �          @�(���\)@�����H�֣�B�����\)@j=q�y���:�RB���                                    Bx����  �          @�  ��{@�G��33��Q�Bӏ\��{@n�R�r�\�.�HB���                                    Bx���~  �          @��\��{@���
=��ffB�33��{@qG��w��/�HBۏ\                                    Bx�� $  T          @����p�@��H�   ��ffB�W
��p�@|(����\�0p�B�\                                    Bx���  �          @����Q�@�(���
����B�k���Q�@����y���&��B�=q                                    Bx��p  �          @�{��@�p�����B��Ϳ�@dz��w
=�.Q�B�{                                    Bx��,  �          @���
�H@��\�!G���B���
�H@]p��z=q�/33B�{                                    Bx��:�  �          @�p���{@�ff�p����B�33��{@g
=�x���/=qB��                                    Bx��Ib  �          @�����@����z���B�33���@~�R�xQ��-G�B�33                                    Bx��X  �          @�{���@���33����B�\)���@�33�x���,��B���                                    Bx��f�  �          @�ff�\@�z������
=B�ff�\@����~{�1Q�B���                                    Bx��uT  �          @�p��G�@�(�������
B�aH�G�@��H�u�*�RBɸR                                    Bx����  �          @��ͿW
=@�����=qBƅ�W
=@�p��l(��#�B���                                    Bx����  �          @�  ��=q@��ÿ�(���ffB�33��=q@���XQ��{B�z�                                    Bx���F  �          @��\>\)@��
��p��E�B�=q>\)@�z��A����
B���                                    Bx����  �          @�녾��R@����˅�33B�L;��R@��R�U���B�k�                                    Bx����  �          @�(�?�G�@�������9B���?�G�@�\)�6ff��=qB��
                                    Bx���8  �          @���@p�@���>���@VffB��@p�@�������^�RB|p�                                    Bx����  �          @��
?Tz�@��\�0��� ��B�z�?Tz�@��\�\)�׮B�L�                                    Bx���  �          @��H�Z=q@P  �Vff���Ck��Z=q?�(���\)�<�C�                                    Bx���*  �          @�z��e@N{�W
=���C
#��e?�Q���\)�8G�C��                                    Bx���  �          @�=q�c�
@U�HQ��C���c�
@Q������1Q�C�                                    Bx��v  �          @�Q��@  @n�R�H���{C �{�@  @   ��{�<  C:�                                    Bx��%  �          @�ff�8Q�@n�R�K��
=B�u��8Q�@\)��
=�?��C:�                                    Bx��3�  �          @��
�
=@�z��ff���
B����
=@mp��`  �(�B��f                                    Bx��Bh  �          @��
��R@����(���{B�z���R@dz��c33�B���                                    Bx��Q  T          @����(��@�Q��%�ڣ�B��(��@L(��u�+{C��                                    Bx��_�  �          @����-p�@��
�1���\)B����-p�@@  �}p��1�
C�                                    Bx��nZ  �          @�p��&ff@�=q�>�R��33B���&ff@8����(��;
=C�3                                    Bx��}   �          @�z���@\)�P���G�B�\)��@.�R���
�K�C��                                    Bx����  �          @�p���\@����P  �z�B�aH��\@8Q���p��M\)B���                                    Bx���L  �          @�p����@�z��Fff��HB�z���@J=q��33�H�\B�8R                                    Bx����  �          @�
=�33@����K���B���33@:=q���H�E��C B�                                    Bx����  �          @�ff��R@���S�
��B�u���R@2�\���L�C �
                                    Bx���>  �          @�
=��@�=q�W
=�{B�p���@1���\)�N�
C )                                    Bx����  �          @�(��$z�@p���Q��{B�Ǯ�$z�@ ������H�\C��                                    Bx���  �          @�{�G�@@����=q�G��B��q�G�?�p������~ffC                                    Bx���0  �          @�33�@�(��8Q����
B�\�@@  �����:�HB��)                                    Bx�� �  �          @����@�33��R��33B�8R��@U��n�R�)�\B���                                    Bx��|  T          @��\��R@�G��#33��{B�.��R@P���q��*�RB���                                    Bx��"  �          @���*�H@���.{���B��*�H@A��xQ��/ffCO\                                    Bx��,�  �          @��H�)��@���$z���(�B�\�)��@H���p  �)��C8R                                    Bx��;n  �          @��\��@�Q��$z����B�\��@^{�w
=�/\)B�                                    Bx��J  �          @��\�{@�(��(�����B�R�{@U��xQ��0z�B�\)                                    Bx��X�  �          @�z��@���)�����B�{�@S�
�xQ��.��B��                                    Bx��g`  �          @�\)�`��@j=q�$z��ָRC޸�`��@*�H�e��G�C�=                                    Bx��v  �          @���6ff@l(��5��ffB�ff�6ff@'��vff�3=qC	p�                                    Bx����  �          @�\)�xQ�@5�Fff�
=C�=�xQ�?޸R�u��'��C�                                    Bx���R  �          @�=q�u@8Q��QG����C&f�u?�(���  �.=qC�f                                    Bx����  �          @��H����@0  �L(��  C������?�\)�x���'��C{                                    Bx����  �          @�{���@(��A����C�
���?���c�
���C&�                                    Bx���D  �          @���=q@33�1��߅C���=q?���Q��33C'@                                     Bx����  
�          @�ff����?����
���
C%33����?(��(Q����HC-h�                                    Bx��ܐ  �          @������?�G���(���{C*����>�(��������C/�                                     Bx���6  �          @������?�p���p���=qC'������?#�
�G���\)C-��                                    Bx����  �          @�{����?�  ��(�����C'������?(�������C-xR                                    Bx���  �          @Å���?�\)������
=C&c����?=p��
=q��p�C,�                                     Bx��(  �          @\���?�33� �����HC#E���?s33��H��  C*&f                                    Bx��%�  �          @��H����?�{�Q�����C �q����?�\)�&ff�ɮC(G�                                    Bx��4t  �          @�=q��{?��ÿ�(�����C!z���{?�������C(8R                                    Bx��C  �          @�=q���
?�\)�#33��ffC����
?�  �@  ���HC(�f                                    Bx��Q�  �          @�
=��=q@   ����z�Cu���=q?��:�H���HC'�                                    Bx��`f  �          @�G���ff@ ����
��(�C�H��ff?��H�3�
��(�C&�H                                    Bx��o  �          @������R@z��p����CT{���R?�ff�.�R�י�C%�                                    Bx��}�  �          @�������?�Q������C�f����?����'
=��\)C'0�                                    Bx���X  �          @�����H?��
�H���C �q���H?����'���p�C(^�                                    Bx����  �          @������?�  �p���Q�C$E���?.{�333�ܸRC,��                                    Bx����  �          @�G���?����ff��\)C'����>�G��&ff��Q�C/c�                                    Bx���J  �          @�����{?8Q��#�
���C,p���{���
�)����(�C4�
                                    Bx����  �          @�33��
==����,(���C2���
=�5�&ff��  C;h�                                    Bx��Ֆ  �          @��H���?c�
��R���C*Ǯ���=�G��(Q��̣�C2ٚ                                    Bx���<  �          @�=q��\)?   �"�\��  C.����\)��\)�$z�����C6�3                                    Bx����  �          @�33���?˅�(����HC#xR���?J=q�3�
���C+��                                    Bx� �  �          @�Q���  ?�33�$z��ʣ�C%���  ?��7���Q�C-��                                    Bx� .  �          @�����Q�?�����R�\C#\��Q�?J=q�6ff��\)C+s3                                    Bx� �  �          @�Q����R?�33�\)��{C"s3���R?W
=�7���(�C*�f                                    Bx� -z  �          @������?�G��)����ffC#������?(���>�R���HC,�q                                    Bx� <   �          @�����z�?�=q�"�\��\)C c���z�?�  �=p���  C(�q                                    Bx� J�  �          @�����@��*�H����Cu����?���L(���  C%�                                     Bx� Yl  �          @�\)���@�\�'���{C� ���?��K�� �\C#��                                    Bx� h  �          @�\)��z�@�
�$z���G�C����z�?��H�HQ����RC#ff                                    Bx� v�  �          @�����\@  �"�\�ˮC{���\?�33�E��{C#��                                   Bx� �^  �          @�����Q�?�(��*�H��
=C���Q�?����HQ���33C'�
                                   Bx� �  �          @�(���
=?У��+���\)C!�{��
=?E��B�\��p�C+#�                                    Bx� ��  �          @�����?�\)�.�R��(�C$�
���?�\�@�����C.33                                    Bx� �P  �          @��R���\?��K���C#�����\>�(��\����C.�f                                    Bx� ��  �          @�ff���
?���N�R�p�CǮ���
?^�R�h�����C)\)                                    Bx� Μ  �          @�
=��p�@33�E��CT{��p�?���c33�=qC'0�                                    Bx� �B  �          @�Q���
=@��E�����CJ=��
=?���c33���C&�R                                    Bx� ��  �          @�����@p��Dz����C� ��?�(��e����C%Y�                                    Bx� ��  T          @Å��  @�R�S33��RC+���  ?��w��p�C"}q                                    Bx�	4  �          @��H���
@%��XQ���CxR���
?��R�~�R�#  C!#�                                    Bx��  �          @��H��ff@7
=�W
=�C�q��ff?�\��G��&�C)                                    Bx�&�  �          @��
���@.�R�^{�	�HC=q���?�\)��33�(ffC)                                    Bx�5&  �          @�G���=q@#�
�U�p�CQ���=q?�  �|(��"�
C ��                                    Bx�C�  �          @��
��\)@\)�Vff���C����\)?��z�H�G�C"\)                                    Bx�Rr  �          @�  ����@+��e���C�3����?�����R�(G�C                                     Bx�a  �          @�p���z�?��H�c33�C���z�?\(��}p�� {C)xR                                    Bx�o�  �          @�����@ff�j=q��C�����?��H���&=qC%)                                    Bx�~d  �          @�{����@2�\�h���	p�C\)����?�������&�C 
                                    Bx��
  �          @�\)���H@+��c�
���CJ=���H?Ǯ���(��C :�                                    Bx���  �          @�\)��  @0���g
=���C����  ?�\)����,(�C
                                    Bx��V  �          @�33���\@5��l���Q�C�H���\?�z���33�,�\C                                      Bx���  T          @ʏ\��z�@=q�c33�C����z�?�����H�!(�C$G�                                    Bx�Ǣ  T          @�z���=q@(Q��g
=�	�RC!H��=q?�  ���R�$�RC!ٚ                                    Bx��H  �          @�����@5�j=q��C@ ���?�Q�����)p�C�                                    Bx���  �          @���z�@G��r�\���C��z�?�
=��Q��2��C�                                    Bx��  �          @�����@Tz��p  ��C������@Q������3ffC.                                    Bx�:  �          @�{��33@J=q�u�Q�CY���33?��H��=q�4��CxR                                    Bx��  �          @�
=�xQ�@L(����\�p�C�{�xQ�?�����@�C�                                    Bx��  �          @�z��s�
@I�������Cc��s�
?�������AQ�C�
                                    Bx�.,  �          @�ff�p��@S33��Q��C
�R�p��@33�����@Q�CaH                                    Bx�<�  �          @�ff����@Mp��n�R���CG�����@�\��\)�0=qCǮ                                    Bx�Kx  �          @Ϯ��G�@Fff�����(�C}q��G�?�{��
=�:�CG�                                    Bx�Z  �          @�����
@R�\�b�\��RC����
@
�H��=q�%�C��                                    Bx�h�  T          @�����  @G
=�����\)C��  ?�\)��Q��6�CG�                                    Bx�wj  �          @�z����@B�\�o\)�	G�CJ=���?����{�'�
C��                                    Bx��  �          @Ӆ���R@@���u��HC�R���R?������,�C��                                    Bx���  �          @��
��(�@E�xQ���C���(�?�33���\�/G�C�                                    Bx��\  �          @��
���@B�\�l(���
CL����?�z���z��&\)CQ�                                    Bx��  �          @�z���33@AG���  �=qC8R��33?����333CxR                                    Bx���  �          @��H��=q@@  �}p���
C8R��=q?�ff��(��2��Ch�                                    Bx��N  �          @�33��@@  �vff���C�f��?�=q�����-G�C��                                    Bx���  �          @��H���H@<(��~�R��
C�)���H?޸R��z��3  C(�                                    Bx��  �          @ҏ\����@<���z=q��
C&f����?�\��=q�/�
C�                                    Bx��@  �          @����  @C�
�k��=qC�=��  ?�Q����
�'
=C�3                                    Bx�	�  �          @љ���=q@E�s�
���Cff��=q?�Q���Q��.p�Cٚ                                    Bx��  
�          @�G���{@W
=�l(��

=C:���{@�R��
=�-(�C�3                                    Bx�'2  T          @�����Q�@S33�k��	\)C@ ��Q�@���{�+p�C��                                    Bx�5�  T          @�  ��@G
=�w��G�C^���?�Q���=q�2�
C{                                    Bx�D~  �          @�\)���R@7��n{�33C33���R?�G�����)33Cz�                                    Bx�S$  �          @�����Q�@0���s�
�  C����Q�?У�����*p�C 33                                    Bx�a�  �          @�=q��Q�@:=q�r�\��C8R��Q�?�\���)��C�=                                    Bx�pp  �          @�G�����@>{�j=q�G�C�=����?�\)���\�%C�
                                    Bx�  �          @У�����@<���hQ���C������?�{��G��%  C��                                    Bx���  �          @�33���
@1G����\�{C�����
?�����p��4��C G�                                    Bx��b  �          @��
��
=@:=q�z=q��RC���
=?�G���G��-�\C�=                                    Bx��  �          @�33��  @?\)�Z�H��ffC�{��  ?�(����H�  C�=                                    Bx���  �          @�33��
=@?\)�aG�� �
C�{��
=?�Q���{��C�)                                    Bx��T  �          @љ���p�@5�P������C�R��p�?�\)�y���=qC33                                    Bx���  �          @У�����@3�
�X�����HC������?������\)C\)                                    Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��F   T          @Ϯ��ff@3�
�s33�ffC� ��ff?�Q�����+G�C(�                                    Bx��  �          @У�����@.�R�~{��HC8R����?��������0��C T{                                    Bx��  �          @�����
=@1��w
=�33C!H��
=?�33���R�,�C��                                    Bx� 8  �          @�\)��ff@:�H�mp���\C����ff?�=q��33�(�\C�)                                   Bx�.�  �          @�z����@Q��\)��C�H���?�  ����/{C'xR                                   Bx�=�  �          @�����@z��z�H��C�R����?u���\�)�C(O\                                    Bx�L*  �          @�ff�vff@p  �S�
��G�C�q�vff@/\)���&�C��                                    Bx�Z�  �          @�=q�W�@�Q��@  ���
B��{�W�@s33�����  C�\                                    Bx�iv  �          @��H�x��@y���g
=�
=C��x��@3�
��Q��,�C&f                                    Bx�x  �          @�Q����\@e�c�
��HC
�����\@!���(��)z�C8R                                    Bx���  �          @أ�����@X����(��\)C����@(����
�8Q�C8R                                    Bx��h  T          @�(��hQ�@���@����{B����hQ�@mp���(��
=C^�                                    Bx��  �          @�
=�y��@����C�
��=qC�H�y��@e��z��  C	ff                                    Bx���  �          @�\)�~�R@��\�7���ffC��~�R@j�H�~{���C	@                                     Bx��Z  �          @����|��@�33�>�R���C�f�|��@j�H���\�ffC	(�                                    Bx��   �          @�\)�~�R@����>{�хC5��~�R@g
=�����{C	�                                    Bx�ަ  �          @��qG�@�z��=p�����C{�qG�@mp���=q�Q�Cu�                                    Bx��L  �          @ָR�~{@�z��333���HC�{�~{@p  �z=q�Q�C�f                                    Bx���  �          @�G����@��\�,(���Q�C  ���@_\)�n{��
C&f                                    Bx�
�  �          @ٙ���=q@����+���\)C����=q@[��l(���\C\                                    Bx�>  �          @�����=q@���=p����HC)��=q@X���~{�p�C�H                                    Bx�'�  �          @��H��z�@����A���z�C�R��z�@Vff������C�f                                    Bx�6�  �          @�
=��z�@�
=�B�\�Σ�C	����z�@R�\�����{C��                                    Bx�E0  �          @�=q��(�@5�b�\��\)C&f��(�?�=q������\C!0�                                    Bx�S�  T          @�{����?����`������C'�����>�(��p  ����C/�                                    Bx�b|  �          @�\��{>L���\����C28R��{�333�X����
=C:+�                                    Bx�q"  �          @�z���Q�@=q�O\)��33CJ=��Q�?�  �p  � �
C%p�                                    Bx��  �          @�z���{@�ff�J=q��z�C����{@P  ��(���\C�{                                    Bx��n  �          @�(���
=@�\)�Z=q��  C���
=@Mp���(��ffC�R                                    Bx��  �          @�����  @�{�]p���
=CaH��  @J=q��p��z�CQ�                                    Bx���  �          @�z�����@z=q�X����{C5�����@:=q�������CG�                                    Bx��`  �          @�����=q@g
=�l����\C����=q@#33��  �"Cٚ                                    Bx��  �          @�(���\)@����Mp��߮C	�
��\)@Fff��z���HCL�                                    Bx�׬  �          @�(���G�@~{�:=q��Q�CY���G�@Fff�tz��\)C!H                                    Bx��R  �          @����
=@]p��QG���33C.��
=@ �������
=C8R                                    Bx���  �          @�ff���H@N{�]p���
=C����H@�R����p�CQ�                                    Bx��  T          @�{��  @QG��E��  C{��  @Q��tz����C��                                    Bx�D  �          @�Q����H@I���L(���{Cu����H@\)�x�����CJ=                                    Bx� �  �          @�����z�@U�Z�H��\)C��z�@
=����=qC\)                                    Bx�/�  �          @�\��@Z�H�Y�����
C����@(������
C�=                                    Bx�>6  �          @������@)���n�R���RCL�����?�\)�������C#�
                                    Bx�L�  �          @�p����H@
�H�s33��C� ���H?����\)�33C(xR                                   Bx�[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�j(   T          @�=q���@G��s33��HC 0����?�  ���{C)�                                    Bx�x�  �          @�(���=q?�(��u��Q�C%�)��=q>�ff���\��
C/��                                    Bx��t  �          @�\���?����mp���\)C"aH���?Tz�������
C+�=                                    Bx��  �          @�����H?ٙ��p�����C#�\���H?333����
z�C-33                                    Bx���  �          @�ff���?��H�{��=qC&
=���>�
=��p��z�C/�H                                    Bx��f  �          @�����  ?\�~{��RC%(���  >���
=��RC/=q                                    Bx��  T          @����z�?������R�{C&�
��z�>aG�������C1�                                    Bx�в  �          @�p����?�Q���{�ffC%�\���>�33������C0s3                                    Bx��X  �          @���=q?.{�qG���C-����=q��\)�tz�� ��C6��                                    Bx���  �          @����Q�?�{�|���C&����Q�>�������=qC0�3                                    Bx���  �          @����(�?W
=�\)�p�C+�H��(��8Q���=q�
�C5��                                    Bx�J  �          @�p���>�Q��\)���C0����&ff�}p��33C::�                                    Bx��  �          @�����>L�����H�
=C2����ͿQ������  C;�3                                    Bx�(�  �          @�\�����G����
�C5\��������~{��\C?#�                                    Bx�7<  �          @�(���z�    �}p����C3����z�z�H�u��
C=xR                                    Bx�E�  �          @������;��\)��
C8u����Ϳ�Q��p  ��G�CA��                                    Bx�T�  �          @�����
�Q���Q��p�C;�����
����k����
CE                                      Bx�c.  �          @�{���
���
�g
=���
C=�
���
��33�O\)����CE@                                     Bx�q�  �          @�  ��?���h����=qC$�{��?
=�z=q�	�RC.�                                    Bx��z  T          @޸R����@#33�W
=��G�C8R����?���x���
=C#}q                                    Bx��   �          @���\)@�
�a����
C!H��\)?�{��  �33C&�                                    Bx���  �          @������?�\�p  �z�C"@ ����?J=q��=q�ffC+��                                    Bx��l  �          @���G�?���}p��  C(aH��G�=����
��C2�R                                    Bx��  �          @޸R��\)?�����G��G�C%�
��\)>�{��Q��(�C0p�                                    Bx�ɸ  �          @����R@p��i����(�C����R?��R���\�\)C'.                                    Bx��^  �          @����@���tz����C@ ���?�����\)���C(�                                    Bx��  �          @�  ����?�\)�u���HC!W
����?^�R�����C+
                                    Bx���  �          @�Q���z�?G����\��C,!H��zᾀ  �����\)C6�\                                   Bx�P  �          @޸R��\)?h�������C*�\��\)����������C5�                                   Bx��  �          @޸R��  ?����(����C)���  <#�
��Q��(�C3�)                                   Bx�!�  �          @�\)���?W
=��(��C+���������ff��
C6�=                                   Bx�0B  �          @����33?k��p  �G�C+���33���w
=���C4E                                    Bx�>�  T          @�����?Tz��h����Q�C,{�����Q��n�R��p�C4��                                    Bx�M�  �          @������\?&ff�w����C-�����\���
�z=q�G�C7�                                    Bx�\4  �          @�ff��=q?E����\���C,
��=q��  ��z��z�C6��                                    Bx�j�  �          @�\)���>�(��i����
=C/޸������i�����\C8�f                                    Bx�y�  �          @�\)���H�#�
�r�\�  C4����H�p���j�H���RC=&f                                    Bx��&  �          @�p���
=�Ǯ�w��G�C7�3��
=��=q�j=q����CA�                                    Bx���  �          @ۅ���H�����}p��
=C7T{���H���
�p�����C@�                                    Bx��r  �          @ۅ��\)>�z��n{��C1���\)�!G��k��=qC:B�                                    Bx��  �          @�p����
=�G��g
=����C2�3���
�G��a���Q�C;��                                    Bx�¾  �          @�����ff���
�u��\)C7:���ff��  �hQ���G�C@W
                                    Bx��d  �          @�
=��G�?��
��33�(�C)z���G�    ��
=�ffC4�                                    Bx��
  �          @�Q�����>��
�z=q�=qC0�=���ÿ#�
�w���C:Q�                                    Bx��  �          @�����녾��
��z���RC7#���녿����|(���
C@Ǯ                                    Bx��V  �          @�=q��    ��ff��C4������
��=q���C>:�                                    Bx�	�  �          @�G���
=>�����33�Q�C0�f��
=�+�������C:��                                    Bx�	�  �          @�����  ���~�R�
Q�C9E��  ��p��n�R� ��CBu�                                    Bx�	)H  �          @�\)����=������\�#��C2����׿��\���R�{C>�                                    Bx�	7�  �          @߮��
=�#�
��(��&\)C4c���
=��z�������C@��                                    Bx�	F�  �          @�  ���\?\)�����+��C-�R���\��R��Q��+��C:�f                                    Bx�	U:  �          @�����;�ff���R��
C8�R���Ϳ�G���
=���CC��                                    Bx�	c�  �          @����(�������
=��HC7�\��(���z���Q���HCB�3                                    Bx�	r�  �          @�\)���>W
=������C1�{����G��\)��
C;��                                    Bx�	�,  T          @߮��=q>�=q��\)��C1B���=q�E�����(�C;�
                                    Bx�	��  �          @������=��
��G����C3������u���ffC=Ǯ                                    Bx�	�x  �          @޸R����=�Q������"��C3\���ÿ�G���p���C>޸                                    Bx�	�  �          @ҏ\���H�&ff�mp��
\)C:�����H���
�\(���=qCD                                      Bx�	��  �          @Ӆ���Ϳ����fff���C?����Ϳ�Q��N{����CG�                                     Bx�	�j  �          @�p���\)�&ff��=q�C;{��\)��\)�r�\�
�HCE(�                                    Bx�	�  �          @�33��
=�����w���HC?����
=��(��`  � 33CH�3                                    Bx�	�  �          @ڏ\���n{��33�G�C>.�������\)�=qCH��                                    Bx�	�\  �          @�\)��(��333��Q���RC;���(����H�}p����CFk�                                    Bx�
  �          @׮���ÿ�\��{�%{C9�q���ÿǮ����
CE@                                    Bx�
�  �          @�  ���H�����\�=qC9
���H�����vff�Q�CC+�                                   Bx�
"N  �          @�\)���H�333�k����C;+����H�����Y�����HCC��                                    Bx�
0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�
?�   S          @�����)���Z�H���CLǮ����X���+�����CR�)                                    Bx�
N@  �          @�\)����O\)�^{����C;������У��J�H�أ�CC0�                                    Bx�
\�  T          @�\)��p�>k���z��'�C1}q��p��fff�����#�C=�)                                    Bx�
k�  �          @����z�>L����Q��+  C1���z�s33����&�C>}q                                    Bx�
z2  �          @����\)=�������&C2�f��\)���
�����!�HC?
                                    Bx�
��  �          @޸R���@�\��33�C�f���?�p������"�C&s3                                    Bx�
�~  �          @�\)��33@{��ff��C� ��33?������%�\C'aH                                    Bx�
�$  �          @�\)���H?��
��{�G�C$���H>����{��\C.�                                    Bx�
��  �          @�{��
=@�
��(����Cs3��
=?s33��  �,C)33                                    Bx�
�p  �          @�ff���?����33�%��C�\���?8Q���p��433C+�\                                    Bx�
�  �          @�{���?�(����H�%��C ����?\)���
�2ffC-p�                                    Bx�
�  �          @���ff?�Q���G��#�HC!.��ff?
=q����0�C-��                                    Bx�
�b  �          @�ff���?��H��p���HC$=q���>�33��z��'z�C0)                                    Bx�
�  �          @�
=��{?�=q��33�{C#
��{>�����%�C.�                                    Bx��  �          @�\)��{?�����=q�C#�H��{?����H���C.                                      Bx�T  T          @ᙚ��
=?�\)��(��ffC#� ��
=?z���z��ffC-�                                   Bx�)�  �          @���\)@�
���
�"�\CxR��\)?c�
��\)�2ffC)�
                                   Bx�8�  �          @�����@����ff�%Q�Cp�����?p�����\�5��C)�                                    Bx�GF  �          @�=q��?�Q������%
=C�\��?B�\��\)�3�RC+5�                                    Bx�U�  �          @�p����
?������{C&O\���
>.{����%�C2&f                                    Bx�d�  �          @�p����?�  �����!�RC&�����=�����=q�(�RC2�                                    Bx�s8  �          @���z�?��������HC)�\��z�L��������RC4z�                                    Bx���  �          @�
=���H>#�
��\)���C2p����H�\(���z��G�C<ff                                    Bx���  �          @�R���;����(���C6� ���Ϳ��R�|����HC?�f                                    Bx��*  �          @�ff���������33�
�RC6�������
�y���Q�C@33                                    Bx���  �          @����G��=p��~{�ffC:�R��G����j�H��
=CCu�                                    Bx��v  �          @��ȣ׿�G���=q���C=!H�ȣ׿��H�mp���CET{                                    Bx��  �          @���Ϯ�(���z���
C9aH�Ϯ��33���
���\CBB�                                    Bx���  �          @�����p��@  ��33���C:����p����
������{CCu�                                    Bx��h  �          @�ff��=q�L�����\��RC;.��=q����Q����HCD�                                    Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�#    S          @�z���ff�}p�����
=C=
��ff� ���\)���
CE�R                                    Bx�1�  �          @�(��У׿n{�u��C<(��У׿�=q�`  �ڣ�CC�                                    Bx�@L  �          @�33�ҏ\�:�H�mp����C:O\�ҏ\�����Z�H�֏\CA�f                                    Bx�N�  �          @�Q���  �(��l����ffC9aH��  ��p��\(��ڣ�C@޸                                    Bx�]�  �          @����H�:�H�g
=��RC:O\���H��=q�U��хCAu�                                    Bx�l>  
�          @�(���G��333�Y���ә�C9�f��G���  �HQ����C@s3                                    Bx�z�  
�          @�p���(��J=q�Tz��̣�C:����(������A���C@�{                                    Bx���  �          @�ff��G��k��a���{C;�R��G��޸R�L���ď\CBff                                    Bx��0  �          @����=q�h���e��33C;�H��=q��  �P����
=CB�H                                    Bx���  �          @�z���{�@  �e��  C:�f��{�˅�S�
��  CA�f                                    Bx��|  �          @������H�E��s�
��
=C:�����H���aG��ڸRCB.                                    Bx��"  �          @����ָR�0���z=q���HC9�
�ָR��{�hQ���{CAz�                                    Bx���  �          @����{�����{��C7�q��{���H�|�����
C@T{                                    Bx��n  �          @�33��p���Q����R�
ffCEn��p��<���w
=�뙚CM��                                    Bx��  �          @�=q��ff�	������Q�CG���ff�G
=�e��
=CN��                                    Bx���  �          @�33��  �*=q��Q���CK�H��  �g
=�_\)��{CS                                      Bx�`  �          @���ff��(���Q��
�
CE����ff�@  �y����{CM�=                                    Bx�  
�          @�\)�Ǯ������H���CDٚ�Ǯ�<(���  ���HCM@                                     Bx�*�  �          @�\)���������H�z�CG}q�����N{�{���  CO��                                    Bx�9R  �          @�ff�����%������Q�CK�������hQ��x����\CS�\                                    Bx�G�  T          @�p���z�����{�\)CJs3��z��`  �}p���ffCR�                                    Bx�V�  e          A   �ƸR�(�������CIu��ƸR�\(��l(��ۮCP�3                                    Bx�eD  
�          A�����Ϳ�\��ff��CC}q�����:�H��(���G�CL�                                    Bx�s�  T          A����z������(�CCǮ��z��>�R��
=��CL�R                                    Bx���  T          A������\)��(���C@
=���$z������z�CI�\                                    Bx��6  �          A��ʏ\���R�����G�CA8R�ʏ\�*=q��G����CJ�{                                    Bx���  T          A�H��=q��G���G���C?E��=q�(����\�Q�CI!H                                    Bx���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��(   S          A���  <��
���
�HC3�\��  ��{��G��\)C=\)                                    Bx���  �          A�H�ҏ\��
=���H��C7��ҏ\��=q���H���CA�=                                    Bx��t  �          A�\����8Q�������C:k������z�����z�CD�{                                    Bx��  �          A�\���
�   ���\��
C8}q���
��(������33CC
                                    Bx���  
�          A�R��(���  ���H�33C6G���(���p����
���CA�                                    Bx�f  T          Aff��G��k�����Q�C5�q��G�������
�
ffC?�q                                    Bx�  "          A{��=q���������C=J=��=q����Q���33CFT{                                    Bx�#�  "          A��������������)�HC7�������޸R��G�� {CD
=                                    Bx�2X  �          @�G������=p������'��CU����������y����RC^��                                    Bx�@�  "          @���@����p������  Cp��@���Ǯ�%���
Ct:�                                    Bx�O�  �          @���\(���{��G��
=qClz��\(��\�9����(�Cp�                                     Bx�^J  �          @����j=q��Q����R��RCi�
�j=q��ff�Fff����Cn^�                                    Bx�l�  �          @��\�N�R��(����
�ffCo\�N�R��33�J�H���
Cs\                                    Bx�{�  �          @�{�fff��ff���
�
��CkO\�fff�Å�>{��Co}q                                    Bx��<  �          @�33�mp����\���R�=qCi޸�mp����R�6ff���\Cn{                                    Bx���  �          @���x�������~�R� {ChG��x����33�(Q���{Cl^�                                    Bx���  
�          @�Q��fff��ff������CkE�fff�����'���  Co)                                    Bx��.  �          @�(��L(������o\)��(�Cq��L(��У��p����Cs�                                    Bx���  "          @�(��A���Q�����p�CrG��A�����7
=����Cu��                                    Bx��z  �          @��������������C`  ����  �S�
���HCe�
                                    Bx��   
�          @�\)�������=q��Ca���������Q���=qCg�                                    Bx���  �          A (����H��Q���
=��RC^ٚ���H���H�qG���G�CeY�                                    Bx��l  �          A ����ff�{���
=�!��C[���ff���H���H��ffCcO\                                    Bx�  T          A Q����R�U������/G�CW=q���R���\�����
�RC`8R                                    Bx��  �          @��H�����@�����DCW���������
��Q��  Cb��                                    Bx�+^  "          @�{�n{�z������a\)CQ��n{�e���{�?�C_�)                                    Bx�:  
�          @�=q����\)��z��G{CR����w
=���H�%Q�C]�)                                    Bx�H�  �          A�H��G��7���Q�� ��COff��G����\��z��G�CX^�                                    Bx�WP  �          A z���Q��p�����%\)CL���Q��l(���G��	�\CUǮ                                    Bx�e�  "          A ������%���H�0Q�CN������x���������CX�q                                    Bx�t�  �          Ap����
�,(���p��1��CO����
��  ���\�ffCZ�                                    Bx��B  "          A����녿�ff�����0��C@�f����,(������  CL�)                                    Bx���  �          A=q����+����3�C:n����
=���\�&p�CG�f                                    Bx���  �          A ����\)�������=CC(���\)�5�����(=qCPs3                                    Bx��4  "          A{��  ��p��\�ACA0���  �,�����H�-�RCO33                                    Bx���  �          Ap������B�\���H�Q  C<�=��������ff�@�CM#�                                    